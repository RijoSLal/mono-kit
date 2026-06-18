// vibe code alert , i am actively learning rust now so, please verify all features are correctly implemented

use memmap2::{MmapMut, MmapOptions};
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_pyobject::from_pyobject;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::Path;

const PAGE_SIZE: usize = 4096;

// =======================================
// metadata stored for every vector
// =======================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaEntry {
    // vector offset inside mmap file
    offset: usize,

    // soft delete flag
    active: bool,

    // arbitrary python dict
    //
    // examples:
    // {
    //   "text": "...",
    //   "author": "...",
    //   "tags": [...],
    //   "anything": ...
    // }
    meta: Value,
}

// =======================================
// object returned to python
// =======================================
#[pyclass]
struct FullRecord {
    #[pyo3(get)]
    idx: String,

    #[pyo3(get)]
    embedding: Vec<f32>,

    #[pyo3(get)]
    active: bool,

    #[pyo3(get)]
    meta: PyObject,
}

// =======================================
// main index
// =======================================
#[pyclass]
struct MemmapIndex {
    dim: Option<usize>,

    mmap: MmapMut,

    file: std::fs::File,

    dir: String,

    // id -> metadata
    map: HashMap<String, MetaEntry>,

    next_offset: usize,
}

impl MemmapIndex {
    // =======================================
    // metadata file path
    // =======================================
    fn meta_path(&self) -> std::path::PathBuf {
        Path::new(&self.dir).join("meta.bin")
    }

    // =======================================
    // persist mmap + metadata
    // =======================================
    fn persist(&mut self) {
        let _ = self.mmap.flush();

        if let Ok(bytes) = bincode::serialize(&(self.dim, &self.map)) {
            let _ = std::fs::write(self.meta_path(), bytes);
        }
    }

    // =======================================
    // load metadata
    // =======================================
    fn load_meta(&mut self) {
        if let Ok(bytes) = std::fs::read(self.meta_path()) {
            if let Ok((dim, map)) =
                bincode::deserialize::<(
                    Option<usize>,
                    HashMap<String, MetaEntry>,
                )>(&bytes)
            {
                self.dim = dim;

                self.map = map;

                self.next_offset = self
                    .map
                    .values()
                    .map(|v| v.offset)
                    .max()
                    .map(|v| v + 1)
                    .unwrap_or(0);
            }
        }
    }

    // =======================================
    // grow mmap file if needed
    // =======================================
    fn ensure_capacity(&mut self, offset: usize) -> PyResult<()> {
        let dim = match self.dim {
            Some(v) => v,
            None => return Ok(()),
        };

        let required = (offset + 1) * dim * 4;

        if self.mmap.len() < required {
            let new_size = required.next_power_of_two().max(PAGE_SIZE);

            self.file
                .set_len(new_size as u64)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            self.mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&self.file)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            };
        }

        Ok(())
    }

    // =======================================
    // cosine similarity
    // =======================================
    fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot: f32 =
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let na: f32 =
            a.iter().map(|v| v * v).sum::<f32>().sqrt();

        let nb: f32 =
            b.iter().map(|v| v * v).sum::<f32>().sqrt();

        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    // =======================================
    // read vector from mmap
    // =======================================
    fn read_embedding_internal(&self, offset: usize) -> Array1<f32> {
        let dim = self.dim.unwrap();

        let start = offset * dim * 4;

        let mut out = Vec::with_capacity(dim);

        for i in 0..dim {
            let pos = start + i * 4;

            let mut bytes = [0u8; 4];

            bytes.copy_from_slice(&self.mmap[pos..pos + 4]);

            out.push(f32::from_le_bytes(bytes));
        }

        Array1::from(out)
    }
}

#[pymethods]
impl MemmapIndex {
    // =======================================
    // constructor
    // =======================================
    #[new]
    fn new(dir: String) -> PyResult<Self> {
        std::fs::create_dir_all(&dir)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let data_path =
            Path::new(&dir).join("embeddings.dat");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if file
            .metadata()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .len()
            == 0
        {
            file.set_len(PAGE_SIZE as u64)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };

        let mut s = Self {
            dim: None,
            mmap,
            file,
            dir,
            map: HashMap::new(),
            next_offset: 0,
        };

        s.load_meta();

        Ok(s)
    }

    // =======================================
    // insert new vector
    //
    // args:
    // idx        -> unique vector id
    // embedding  -> numpy float32 vector
    // meta       -> arbitrary python dict
    // =======================================
    fn insert(
        &mut self,
        idx: String,
        embedding: PyReadonlyArray1<f32>,
        meta: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let emb = embedding.as_array().to_owned();

        // first insert decides dimension
        if self.dim.is_none() {
            self.dim = Some(emb.len());
        }

        let dim = self.dim.unwrap();

        if emb.len() != dim {
            return Err(PyValueError::new_err(
                "dimension mismatch",
            ));
        }

        // python dict -> serde_json::Value
        let meta_value: Value = from_pyobject(meta)?;

        let offset = self.next_offset;

        self.ensure_capacity(offset)?;

        let start = offset * dim * 4;

        // write vector into mmap
        for (i, v) in emb.iter().enumerate() {
            let pos = start + i * 4;

            self.mmap[pos..pos + 4]
                .copy_from_slice(&v.to_le_bytes());
        }

        // store metadata
        self.map.insert(
            idx,
            MetaEntry {
                offset,
                active: true,
                meta: meta_value,
            },
        );

        self.next_offset += 1;

        self.persist();

        Ok(())
    }

    // =======================================
    // update existing vector + metadata
    // =======================================
    #[pyo3(signature = (idx, embedding, meta=None))]
    fn update(
        &mut self,
        idx: String,
        embedding: PyReadonlyArray1<f32>,
        meta: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let emb = embedding.as_array().to_owned();

        let entry = match self.map.get_mut(&idx) {
            Some(v) => v,
            None => {
                return Err(PyValueError::new_err(
                    "index not found",
                ))
            }
        };

        let dim = self.dim.unwrap();

        if emb.len() != dim {
            return Err(PyValueError::new_err(
                "dimension mismatch",
            ));
        }

        // overwrite metadata if provided
        if let Some(m) = meta {
            let meta_value: Value = from_pyobject(m)?;
            entry.meta = meta_value;
        }

        let start = entry.offset * dim * 4;

        // overwrite vector
        for (i, v) in emb.iter().enumerate() {
            let pos = start + i * 4;

            self.mmap[pos..pos + 4]
                .copy_from_slice(&v.to_le_bytes());
        }

        self.persist();

        Ok(())
    }

    // =======================================
    // soft delete
    // =======================================
    fn delete(&mut self, idx: String) -> PyResult<()> {
        if let Some(v) = self.map.get_mut(&idx) {
            v.active = false;

            self.persist();

            Ok(())
        } else {
            Err(PyValueError::new_err(
                "index not found",
            ))
        }
    }

    // =======================================
    // list active ids
    // =======================================
    fn list_all(&self) -> Vec<String> {
        self.map
            .iter()
            .filter(|(_, v)| v.active)
            .map(|(k, _)| k.clone())
            .collect()
    }

    // =======================================
    // get full records
    // =======================================
    fn get_many(
        &self,
        py: Python,
        indices: Vec<String>,
    ) -> PyResult<Vec<FullRecord>> {
        let mut out = Vec::new();

        for idx in indices {
            let meta = match self.map.get(&idx) {
                Some(v) => v,
                None => continue,
            };

            let emb =
                self.read_embedding_internal(meta.offset);

            // serde_json::Value -> python dict
            let py_meta =
                pythonize::pythonize(py, &meta.meta)?;

            out.push(FullRecord {
                idx,
                embedding: emb.to_vec(),
                active: meta.active,
                meta: py_meta.into(),
            });
        }

        Ok(out)
    }

    // =======================================
    // top-k similarity search
    //
    // args:
    // query       -> query embedding
    // k           -> top results
    // batch_size  -> processing chunk size
    // types       -> list of dtypes to filter by
    // =======================================
    #[pyo3(signature = (query, k, batch_size, types=None))]
    fn topk(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        batch_size: usize,
        types: Option<Vec<String>>,
    ) -> PyResult<Vec<(String, f32)>> {
        let q = query.as_array().to_owned();

        let dim = self.dim.unwrap_or(0);

        if q.len() != dim {
            return Err(PyValueError::new_err(
                "query dimension mismatch",
            ));
        }

        let type_set: Option<std::collections::HashSet<String>> =
            types.map(|v| v.into_iter().collect());

        let entries: Vec<_> = self
            .map
            .iter()
            .filter(|(_, v)| {
                v.active
                    && match &type_set {
                        None => true,
                        Some(set) => {
                            // we need to check the 'dtype' field in meta.
                            // wait, in the new schema, meta is a JSON Value.
                            // The previous schema had a 'dtype' field.
                            // Let's check MetaEntry.
                            if let Some(dtype) = v.meta.get("dtype").and_then(|d| d.as_str()) {
                                set.contains(dtype)
                            } else {
                                // if it's just a string (like in main.py)
                                if let Some(s) = v.meta.as_str() {
                                     set.contains(s)
                                } else {
                                    false
                                }
                            }
                        }
                    }
            })
            .collect();

        let mut results: Vec<(String, f32)> = vec![];

        for chunk in entries.chunks(batch_size) {
            let mut local: Vec<(String, f32)> = vec![];

            for (idx, meta) in chunk {
                let emb =
                    self.read_embedding_internal(meta.offset);

                let sim = Self::cosine(&q, &emb);

                local.push(((*idx).clone(), sim));
            }

            local.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap()
            });

            local.truncate(k);

            results.extend(local);
        }

        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });

        results.truncate(k);

        Ok(results)
    }
}

// =======================================
// python module
// =======================================
#[pymodule]
fn rust_engine(
    _py: Python,
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    m.add_class::<MemmapIndex>()?;
    m.add_class::<FullRecord>()?;

    Ok(())
}