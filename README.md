# Verifier-Independence-RAG

**"Can We Trust the Verifier? Verification Independence in RAG Under Knowledge-Base Poisoning"**

## Project Description

This project investigates whether a separate verifier actually improves the reliability of Retrieval-Augmented Generation (RAG) when the external knowledge used by the system may contain poisoned or misleading information. A standard RAG system retrieves relevant passages and provides them to a language model for answer generation. Recent defenses add filtering, evidence checking, or a second model to verify the generated answer. However, a verifier may still accept the same incorrect answer when it depends on the same poisoned passages, retriever, or information source as the original generator. Using a different language model therefore does not necessarily provide an independent safety check.
