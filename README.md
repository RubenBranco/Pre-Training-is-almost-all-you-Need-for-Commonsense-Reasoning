# Pre-Training is Almost All You Need for Commonsense Reasoning

This is an implementation of the paper ["Pre-Training is (Almost) All You Need: An Application to Commonsense Reasoning"](https://www.aclweb.org/anthology/2020.acl-main.357.pdf).

This implementation was designed to be as efficient as possible (or as efficient as I know how to), to be run in a single RTX Titan GPU. In order to do this, instead of performing a single forward pass per hypothesis, further decomposition is done to perform N forward passes per hypothesis. Even with this decomposition, it still is necessary to filter out examples whose premise is larger than a given value. This is due to the fact that its expensive to maintain a computation graph with the activation values for such a large amount of forward passes.

This is a rudimentary implementation. There maybe lingering bugs somewhere and further optimizations could likely be performed. Feel free to ask for pull requests if you wish to share your bugfixes/optimizations. For those who practice beautiful coding: this code may contain some heart-attack inducing repetitions, my apologies.

It contains an implementation for the CommonsenseQA and Argument Reasoning Comprehension tasks.
