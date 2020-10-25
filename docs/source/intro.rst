Introduction
============

``memento`` is a tool for detecting differential variability (DV) and differential correlction (DC) in single-cell RNA-seq datasets.

Motivation
**********

While many methods exist for the comparison of means, we desired to create a tool to 1) accurate estimate the first and second order moments of gene expression, while considering the noise process, 2) provide a framework to perform hypothesis testing in a semi-parametric manner, and 3) incorporate the existence of natural replicates present in most scRNA-seq data.