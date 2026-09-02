# HOSAG Project — Local Setup Guide

This guide explains how to run the HOSAG Project locally so you can test both the backend (FastAPI + Transformers) and the frontend (React/Vite).

---

## Prerequisites
- Python 3.10+ installed
- Node.js + npm installed
- Git installed
- Virtual environment tool (`venv`)
- Create .env in frontend root folder. "VITE_API_BASE=http://127.0.0.1:8000"

---

##  Backend Setup (FastAPI + Transformers)

1. Open **PowerShell** and activate your virtual environment:
   ```powershell
   cd C:\2026\WITS\ResearchProject\HOSAG_Project Folder
   .\venv\Scripts\activate

2. Install dependencies
   ```powershell
   pip install -r requirements.txt

4. Run the backend (**Keep running**)
   ```powershell
   python inference_server.py

## Frontend Setup (React/Vite)
1. Open a new **PowerShell** terminal, do not activate your virtual environment
   ```powershell
   cd frontend
   npm install
   npm run dev
