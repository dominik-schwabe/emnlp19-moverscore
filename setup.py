#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="moverscore",
    version=0.95,
    description="MoverScore: Evaluating text generation with contextualized embeddings and earth mover distance",
    long_description="MoverScore is a semantic-based evaluation metric for text generation tasks, e.g., machine translation, text summarization, image captioning, question answering and etc, where the system and reference texts are encoded by contextualized word embeddings finetuned on Multi-Natural-Language-Inference, then the Earth Mover Distance is leveraged to compute the semantic distance by comparing two sets of embeddings resp. to the system and reference text",
    url="https://github.com/AIPHES/emnlp19-moverscore",
    author="Wei Zhao",
    author_email="andyweizhao1@gmail.com",
    maintainer_email="andyweizhao1@gmail.com",
    license="Apache License 2.0",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "machine translation, evaluation, NLP, natural language processing, computational linguistics"
    ],
    packages=find_packages(),
    install_requires=[
        "pyemd==0.5.1",
        "pytorch-pretrained-bert",
        "torch>=1.0.0",
        "transformers>=1.1.0",
    ],
    extras_require={},
)
