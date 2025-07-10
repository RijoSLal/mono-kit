from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
desc = "A flexible toolkit for semantic search across text, audio, and image data, featuring customizable embedding model support."

setup(
     name="mono-kit", 
     version="0.1.3",
     packages= find_packages(), 
     description=desc,
     install_requires=[
         
            "chromadb==1.0.13",
            "librosa==0.11.0",
            "tensorflow_hub==0.16.1",
            "tensorflow==2.19.0",
            "semantic_text_splitter==0.27.0",
            "numpy==2.1.3",
            "Pillow==11.2.1",

    ],
    long_description=long_description,
    long_description_content_type="text/markdown",


)