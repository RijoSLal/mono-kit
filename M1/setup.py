from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
desc = "ML toolkit for developers to build document, audio, and image similarity retrieval systems with pretrained and finetunable models—ready to use out of the box."

setup(
     name="mono-kit", 
     version="0.1.4",
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