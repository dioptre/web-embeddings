# Semantic search using llm context embeddings in web worker / js / browser

## Background

If you want to do semantic search in the browser, you can use the GTE-Small model from Hugging Face / ONNX Runtime Web. You can also use other models, but this is a good starting point.

## Model

* [GTE Small](https://huggingface.co/thenlper/gte-small/tree/main/onnx) ONNX model
* 60 MB
* This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.
* Performs better than the [OpenAI text-embedding-ada-002 Embeddings](https://platform.openai.com/docs/guides/embeddings)
* Takes roughly 500ms per embedding

## Build & Run

```sh
nvm use
yarn install
yarn start
```

## Test

* Check the console
