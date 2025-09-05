# Foodly AI Support Agent

A comprehensive AI support system built with LangGraph and Databricks that provides intelligent customer service through specialized tools and agents.

## 🏗️ Project Structure

### 📁 **Tool Creation Files** (1-3)
These files contain the logic for creating specialized AI tools:

- **`1.policy_agent_tool_creation`** - Creates tools for handling policy-related queries (refunds, cancellations, terms of service)
- **`2.orders_agent_tool_creation`** - Creates tools for order management and tracking functionality  
- **`3.escalation_agent_tool_creation`** - Creates tools for escalating complex issues to human agents

Each tool file defines specific capabilities that the main agent can use to handle different types of customer inquiries.

### 🚀 **Entry Point**
- **`main`** - **START HERE** - The main application entry point that orchestrates all tools and agents

### 🔧 **Supporting Files**
- **`5.deploy_agent`** - Deployment and serving configuration for the agent
- **`helpers.py`** - Utility functions and helper methods used across the project

## 🎯 Getting Started

**Begin by examining `main.py`** - this file shows how all the components work together and demonstrates the overall agent architecture.

## 💡 How It Works

1. **Tool Creation**: The first 3 files define specialized tools for different support scenarios
2. **Integration**: `main.py` combines these tools into a unified support agent
3. **Deployment**: The deploy script makes the agent available as a service
4. **Helpers**: Common utilities support the entire system

