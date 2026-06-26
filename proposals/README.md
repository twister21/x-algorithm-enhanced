# X-Algorithm Enhanced: User-Steerable Feed Reference Implementation

These files provide a reference implementation for enhancing the X recommendation system, specifically focusing on **user control**. 

The logic contained here is based on the proposals found in [twitter/the-algorithm#1449](https://github.com/twitter/the-algorithm/issues/1449).

## Overview

Current algorithmic feeds often prioritize engagement metrics over explicit user intent. This sample demonstrates a technical framework to shift that balance, allowing users to steer their feeds using improved filtering and interest-inference logic.

## Key Logic & Architecture

The sample code demonstrates decisions for a recommendation pipeline:

### 1. Inclusive and Exclusive Filtering
The system moves beyond simple "keyword blocking" to a dual-layered filter architecture:
* **Inclusive Filters:** Explicitly prioritize content matching specific user-defined preferences
* **Exclusive Filters:** Hard constraints that prune content from the candidate set

### 2. Triple Matching Strategy
To ensure content relevance, the implementation utilizes three distinct strategies:
* **Exact Matching:** High-precision matching for specific tokens and tags.
* **Semantic Matching (NLP):** Leveraging Natural Language Processing to identify themes and intent that keyword matching might miss.
* **Heuristic Mapping:** Using metadata and user history to infer relevance.

### 3. Temporal Interest Inference
The algorithm differentiates between content age and user interest duration:
* **Interest Windows:** Distinguish between short-term "spikes" (e.g., breaking news) and long-term interests.
* **Content Decay:** Logic for how content age interacts with user-defined steerability weights.

## Technical Implementation Note

### Python as a Blueprint
The code is written in Python to provide a clear, readable reference for the underlying logic.

### Transitioning to Rust
Translating these concepts into the production environment (the official Rust codebase) requires access to internal traits and modules. Specifically, full integration depends on:
* `clients` modules for fetching user-state.
* `params` modules for dynamic configuration.
* `util` modules for shared pipeline helpers.
