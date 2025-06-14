


# MCP Causal Inference Server

A Python-based MCP server implementing causal inference analysis on synthetic customer spending data. Uses DoWhy for causal effect estimation and FastMCP for exposing causal tools. Enables interactive causal analysis with treatment and outcome variables from program signup data.

This repository contains a Python MCP server (`server.py`) that performs causal inference analysis on synthetic customer spending data.

The project uses the FastMCP framework to expose causal analysis tools and DoWhy to estimate causal effects using methods like propensity score matching and linear regression.

## Features

- Synthetic dataset simulating user spending before and after program signup
- Causal graph modeling treatment effects and confounders
- MCP tools for estimating causal effects and querying variable relationships
- Easily extensible for custom causal inference workflows

## Setup

1. Create a Python virtual environment and activate it:
'''bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

2. python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

3. python server.py


##
💡 Project Purpose

This server demonstrates how language-based tool interfaces can support causal analysis workflows using natural language interfaces. The setup is useful for researchers, data scientists, or ML systems that need interpretable causal inference.


