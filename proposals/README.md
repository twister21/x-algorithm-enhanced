# X-Algorithm Enhanced: User-Steerable Feed Reference Implementation

This draft covers a subset of the proposals found in twitter/the-algorithm#1449. It presents user control features designed to elevate the personal experience with the X recommendation system. Functionality regarding other directly related aspects mentioned in the issue, such as content types and length, the reply section, account types, or user activity, is omitted here for brevity.

## Overview

Current algorithmic feeds often prioritize engagement metrics over explicit user intent. This sample demonstrates a technical framework to shift that balance, allowing users to steer their feeds using improved filtering and interest-inference logic.

## Key Logic & Architecture

The code outlines decisions for a recommendation pipeline:

### 1. Inclusive and Exclusive Filtering
The system moves beyond simple "keyword blocking" to a dual-layered filter architecture:
* **Inclusive Filters:** Explicitly prioritize content matching specific user-defined preferences
* **Exclusive Filters:** Hard constraints that prune content from the candidate set

### 2. Triple Matching Strategy
To ensure content relevance, the implementation utilizes three distinct strategies:
* **Hard enum:**   genuinely bounded, exhaustive (graph depth, style, sentiment)
                 → exact match
* **Soft enum:**   bounded vocabulary, but semantic adjacency matters (tone)
                 → enum defines vocabulary; scorer uses cosine similarity
* **Free field:**  open / hierarchical / evolving (topics, formats, account follower range)
                 → embedded at preference-save time; dot-product at score time
                 → OR numeric range for follower count

_NLP_ parsing:
All fields are expressible via natural language. The intent parser maps free text to corresponding structured fields. The client may present shortcut options (chips, dropdowns) as convenience, but these are purely UI-layer suggestions.

### 3. General And Temporal Interest Inference
The algorithm differentiates between content age and user interest duration to properly weigh signaling effects:
* **Interest Windows:** Distinguish between short-term "spikes" (e.g., breaking news) and long-term interests based on metadata and user history
* **Content Decay:** Logic for how content age interacts with user-defined steerability weights.

### 4. More
For explanations of the other design decisions, please review the in-line comments throughout the code. Keep in mind that these concepts are based solely on the developer's assumptions of sensible feature additions and engineering approaches. Consequently, they call for refinement to:
* Align with the current product context and proposed system integration
* Better tailor the framework to specific use cases and practical considerations
* Maximize infrastructure efficiency

Discussions about both the overall strategy and specific details are welcome.

## Technical Implementation Note

### Python as a Blueprint
The code is written in Python to provide a clear, readable reference for the underlying logic.

### Transitioning to Rust
Incorporating these concepts into the production environment (the target Rust codebase) requires access to internal traits and modules. Specifically, full implementation depends on:
* `clients` module for fetching user state.
* `params` module for dynamic configuration.
* `util` module for shared pipeline helpers.
