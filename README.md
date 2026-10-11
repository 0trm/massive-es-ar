# massive-es-ar

Do NLU models trained on es-ES lose accuracy on Rioplatense Spanish?

This repo builds an es-AR test set by rewriting the es-ES split of
[MASSIVE](https://github.com/alexa/massive) (Amazon, 2022) into Rioplatense Spanish, then scores
intent and slot models trained on es-ES against both.

## Setup

```sh
uv sync --all-groups
uv run pytest
```

## Licence

- Code: [MIT](LICENSE).
- Data derived from MASSIVE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like
  MASSIVE itself. Attribution: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset with 51 Typologically-Diverse Languages", ACL 2023,
  [arXiv:2204.08582](https://arxiv.org/abs/2204.08582).
