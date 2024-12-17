# Subnet 1 API Documentation

This document describes the API endpoints available for [Subnet 1](https://github.com/macrocosm-os/prompting)

## Table of Contents
- [Getting Started](#getting-started)
- [API Management](#api-management)
  - [Create API Key](#create-api-key)
  - [Modify API Key](#modify-api-key)
  - [Delete API Key](#delete-api-key)
- [Miner Availabilities](#miner-availabilities)
  - [Get Miner Availabilities](#get-miner-availabilities)
  - [Get Available Miners](#get-available-miners)
- [GPT](#gpt)
  - [Mixture of Agents](#mixture-of-agents)
  - [Proxy Chat Completions](#proxy-chat-completions)
- [Health](#health)

---

## Getting Started

Follow these steps to set up and run the API server:

1. **Install dependencies**: Ensure all required dependencies are installed using Poetry.
2. **Run the API server**: Start the server to access the API endpoints.

Use the following command:

```bash
# Run the API server
bash run_api.sh
```

---

## API Management

### Create API Key

**Endpoint:** `POST /api_management/create-api-key/`

**Description:** Creates a new API key with a specified rate limit.

**Parameters:**

- **rate\_limit** (query, required): The rate limit for the API key (integer).
- **admin-key** (header, required): Admin key for authorization (string) defined by validator in `.env.validator`.

---

### Modify API Key

**Endpoint:** `PUT /api_management/modify-api-key/{api_key}`

**Description:** Modifies the rate limit of an existing API key.

**Parameters:**

- **api\_key** (path, required): The API key to modify (string).
- **rate\_limit** (query, required): The new rate limit for the API key (integer).
- **admin-key** (header, required): Admin key for authorization (string) defined by validator in `.env.validator`.

---

### Delete API Key

**Endpoint:** `DELETE /api_management/delete-api-key/{api_key}`

**Description:** Deletes an existing API key.

**Parameters:**

- **api\_key** (path, required): The API key to delete (string).
- **admin-key** (header, required): Admin key for authorization (string) defined by validator in `.env.validator`.

---

## Miner Availabilities

### Get Miner Availabilities

**Endpoint:** `POST /miner_availabilities/miner_availabilities`

**Description:** Fetches miner availabilities based on provided UIDs.

**Request Body:**

- JSON array of integers or null (optional).

---

### Get Available Miners

**Endpoint:** `GET /miner_availabilities/get_available_miners`

**Description:** Retrieves a list of available miners for a specific task and/or model.

**Parameters:**

- **task** (query, optional): The type of task (e.g., `QuestionAnsweringTask`, `SummarizationTask`, etc.).
- **model** (query, optional): The specific model (string).
- **k** (query, optional): The maximum number of results to return (integer, default: 10).

---

## Chat Endpoints

### Mixture of Agents

**Endpoint:** `POST /mixture_of_agents`

**Description:** Combines multiple agents for a task.

**Parameters:**

- **api-key** (header, required): API key for authorization (string).

---

### Proxy Chat Completions

**Endpoint:** `POST /v1/chat/completions`

**Description:** Proxies chat completions to an underlying GPT model.

**Parameters:**

- **api-key** (header, required): API key for authorization (string).

---

## Health

**Endpoint:** `GET /health`

**Description:** Health check endpoint.

---
