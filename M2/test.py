import torch
import numpy as np
import os
import logging
from mono_v2.main import MonoDB

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_comparative_modalities():
    """
    test suite focusing on specific files and performing comparative similarity checks.
    """
    logger.info("--- starting comparative modality tests ---")
    
    store_dir = "./store"
    if os.path.exists(store_dir):
        import shutil
        shutil.rmtree(store_dir)
    
    db = MonoDB(dir=store_dir)
    
    # define test groups
    test_groups = {
        "dove": {
            "image": ["test_dir/dove1.jpeg", "test_dir/dove2.jpeg"],
            "audio": ["test_dir/dove.mp3", "test_dir/dove2.mp3"],
            "text": ["a bird", "a dove"]
        },
        "diff": {
            "image": ["test_dir/diff.jpeg"],
            "audio": ["test_dir/diff.mp3"],
            "video": ["test_dir/diff.mp4"]
        },
        "cycle": {
            "video": ["test_dir/cycle1.mp4", "test_dir/cycle2.mp4"]
        }
    }

    # --- 1. testing custom model ---
    logger.info(f"\n{'='*20} testing custom {'='*20}")
    custom_embs = {}
    for group, mods in test_groups.items():
        custom_embs[group] = {}
        for mod, paths in mods.items():
            custom_embs[group][mod] = []
            for path in paths:
                try:
                    emb = db.embed(path, mod, inbuilt_model=False)
                    custom_embs[group][mod].append((path, emb))
                    logger.info(f"[custom] {mod}: {path} -> {emb.shape}")
                except Exception as e:
                    logger.error(f"failed to embed {path}: {e}")

    logger.info("\n--- custom group similarities ---")
    for group, mods in custom_embs.items():
        logger.info(f">> group: {group}")
        if "text" in mods and mods["text"]:
            t_path, t_emb = mods["text"][0]
            for m_type, items in mods.items():
                if m_type == "text": continue
                for p, e in items:
                    sim = cosine_similarity(t_emb, e)
                    logger.info(f"   text vs {m_type}: '{t_path}' <-> '{p}' = {sim:.4f}")
        for m_type, items in mods.items():
            if len(items) > 1:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        sim = cosine_similarity(items[i][1], items[j][1])
                        logger.info(f"   {m_type} vs {m_type}: '{items[i][0]}' <-> '{items[j][0]}: {sim:.4f}")

    if "dove" in custom_embs and "diff" in custom_embs:
        d_img = custom_embs["dove"].get("image")
        f_img = custom_embs["diff"].get("image")
        if d_img and f_img:
            sim = cosine_similarity(d_img[0][1], f_img[0][1])
            logger.info(f"\n--- custom mismatch check ---\n   dove vs diff (image): {sim:.4f}")

    # --- 2. testing database operations (via custom embeddings) ---
    logger.info(f"\n{'='*20} testing db operations {'='*20}")
    
    # test insert
    logger.info("testing insert (custom embeddings)...")
    for group, mods in custom_embs.items():
        for mod, items in mods.items():
            for path, emb in items:
                # use path as idx for testing
                db.insert(idx=path, embedding=emb, type=mod, meta={"source": group})
    
    # test list_all
    active_ids = db.list_all()
    logger.info(f"active ids count: {len(active_ids)}")
    assert len(active_ids) > 0, "insert or list_all failed"

    # test topk search
    if "dove" in custom_embs and custom_embs["dove"]["image"]:
        query_path, query_emb = custom_embs["dove"]["image"][0]
        logger.info(f"testing topk search for: {query_path}")
        results = db.topk(embedding=query_emb, k=3, batch_size=10, type="image")
        for idx, sim in results:
            logger.info(f"   topk match: {idx} (sim: {sim:.4f})")
        assert results[0][0] == query_path, "topk search failed to find query image"

    # test update
    if active_ids:
        target_id = active_ids[0]
        logger.info(f"testing update for: {target_id}")
        # fetch dummy embedding to overwrite
        dummy_emb = np.random.randn(1024).astype(np.float32)
        db.update(idx=target_id, embedding=dummy_emb, type="text", meta={"updated": True})
        logger.info("update complete")

    # test delete
    if len(active_ids) > 1:
        delete_id = active_ids[1]
        logger.info(f"testing delete for: {delete_id}")
        db.delete(delete_id)
        assert delete_id not in db.list_all(), "delete failed"

    # --- 3. testing imagebind model ---
    logger.info(f"\n{'='*20} testing imagebind {'='*20}")
    ib_embs = {}
    imagebind_available = True
    for group, mods in test_groups.items():
        if not imagebind_available: break
        ib_embs[group] = {}
        for mod, paths in mods.items():
            if not imagebind_available: break
            ib_embs[group][mod] = []
            for path in paths:
                try:
                    emb = db.embed(path, mod, inbuilt_model=True)
                    ib_embs[group][mod].append((path, emb))
                    logger.info(f"[imagebind] {mod}: {path} -> {emb.shape}")
                except Exception as e:
                    logger.error(f"ImageBind test failed, skipping remaining ImageBind tests: {e}")
                    imagebind_available = False
                    break

    if imagebind_available:
        logger.info("\n--- imagebind group similarities ---")
        for group, mods in ib_embs.items():
            logger.info(f">> group: {group}")
            if "text" in mods and mods["text"]:
                t_path, t_emb = mods["text"][0]
                for m_type, items in mods.items():
                    if m_type == "text": continue
                    for p, e in items:
                        sim = cosine_similarity(t_emb, e)
                        logger.info(f"   text vs {m_type}: '{t_path}' <-> '{p}' = {sim:.4f}")
            for m_type, items in mods.items():
                if len(items) > 1:
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            sim = cosine_similarity(items[i][1], items[j][1])
                            logger.info(f"   {m_type} vs {m_type}: '{items[i][0]}' <-> '{items[j][0]}': {sim:.4f}")

        if "dove" in ib_embs and "diff" in ib_embs:
            d_img = ib_embs["dove"].get("image")
            f_img = ib_embs["diff"].get("image")
            if d_img and f_img:
                sim = cosine_similarity(d_img[0][1], f_img[0][1])
                logger.info(f"\n--- imagebind mismatch check ---\n   dove vs diff (image): {sim:.4f}")
    else:
        logger.warning("ImageBind tests were skipped due to loading failure.")

    logger.info("\n--- tests completed! ---")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    test_comparative_modalities()
