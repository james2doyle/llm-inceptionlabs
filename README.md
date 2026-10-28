# llm-inceptionlabs
[![Tests](https://github.com/james2doyle/llm-inceptionlabs/actions/workflows/test.yml/badge.svg)](https://github.com/james2doyle/llm-inceptionlabs/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/james2doyle/llm-inceptionlabs/blob/main/LICENSE)

Access [Inception](https://www.inceptionlabs.ai/) Mercury models via API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install 'https://github.com/james2doyle/llm-inceptionlabs/archive/refs/heads/main.zip'
```

Or directly with pip:

```bash
pip install git+https://github.com/james2doyle/llm-inceptionlabs
```

## Usage

Obtain an [Inception API key](https://platform.inceptionlabs.ai/dashboard/api-keys) and save it like this:

```bash
llm keys set inception
# <Paste key here>
```

Or export it as an environment variable:

```bash
export INCEPTION_API_KEY=...
```

Run `llm models` to get a list of models. Model IDs are namespaced with an
`inceptionlabs/` prefix to avoid conflicts with models from other plugins,
e.g. `inceptionlabs/mercury-3`. Three models are included:
`inceptionlabs/mercury-3`, `inceptionlabs/mercury-2.5` and
`inceptionlabs/mercury-2`.

Run prompts like this:

```bash
llm -m inceptionlabs/mercury-3 'What is a diffusion model?'
```

Reasoning effort accepts `instant`, `low`, `medium` or `high`:

```bash
llm -m inceptionlabs/mercury-3 'What is a diffusion model?' -o reasoning_effort instant
```

Mercury models are diffusion LLMs — set `diffusing` to stream the intermediate
denoising steps instead of just the final output. Note that in this mode each
chunk contains the full text as it is refined, not a delta:

```bash
llm -m inceptionlabs/mercury-3 'What is a diffusion model?' -o diffusing true
```

Note: `mercury-3` accepts temperatures between 0.5 and 1.0.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-inceptionlabs
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```
