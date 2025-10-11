Geometric Primitive-Based Shape Fitting for Scene Objects.
This document describes how to prepare the model, set up the Docker environment, and run example scripts.

---

## 🧩 1. Model Preparation

Download the SAM model checkpoint from the official link:  
👉 [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Then, place the file into the `models/` directory of this repository:

```bash
models/
└── sam_vit_h_4b8939.pth
```

---

## 🐳 2. Run with Docker

### Step 1: Enter Docker Configuration Folder

```bash
cd .docker
```

### Step 2: Launch Container

Run the following command, replacing the path with your actual local path to GPBSF:

```bash
PATH_TO_GPBSF=/absolute/path/to/gpbsf docker compose run gpu
```

> 💡 **Tip:**  
> Make sure Docker and NVIDIA Container Toolkit are properly installed to enable GPU access.

---

## 🚀 3. Run Examples

After entering the container (or on your host system if dependencies are installed):

```bash
cd ~/code/gpbsf
python main.py
```

---

## ⚙️ 4. Notes

- Ensure the `models/` directory exists before running.  
- Always use an **absolute path** for `PATH_TO_GPBSF` when running Docker.

---
