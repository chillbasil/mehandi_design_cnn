# Mehandi Design Multi-Label CNN

A deep learning project that trains a **Convolutional Neural Network (CNN)** to classify **Mehandi (Henna) design patterns** such as:

 Flower &nbsp; Leaf &nbsp; Lotus &nbsp; Mango &nbsp; Peacock  

The dataset was exported from **[Roboflow](https://universe.roboflow.com)** and contains labeled images for **training**, **validation**, and **testing**.

---

## 1. Setup

### Install Dependencies

Make sure you have **Python 3.10+** and **pip** installed, then run:

```bash
pip install tensorflow pandas numpy matplotlib
```

---

## 2. Train the Model

### Run the Training Script

Open a terminal or command prompt in your project directory and run:

```bash
python mehandi_design_multilabel_cnn.py
```

### What the Script Does
- Loads images and labels from dataset folders (`train`, `valid`, `test`)  
- Builds and trains a CNN model  
- Displays **training accuracy** and **loss graphs**  
- Saves the trained model as **`mehandi_multilabel_cnn.h5`**

---

## 3. Predict a Single Image

### Steps to Test an Image

1. Open **`mehandi_design_multilabel_cnn.py`**  
2. Scroll to the bottom and find this line:

   ```python
   # predict_single_image(model, "path_to_image.jpg")
   ```

3. Uncomment it (remove the `#`) and replace `"path_to_image.jpg"` with your actual image path:

   ```python
   predict_single_image(model, "test_image.jpg")
   ```

4. Re-run the script:

   ```bash
   python mehandi_design_multilabel_cnn.py
   ```

You’ll see your image displayed with its **predicted design labels** (e.g., “flower, leaf”).

---

## 4. Project Structure

```text
Mehandi_Designs/
├── mehandi_design_multilabel_cnn.py
├── train/
│   └── _classes.csv
├── valid/
│   └── _classes.csv
├── test/
│   └── _classes.csv
└── mehandi_multilabel_cnn.h5
```

---

## 5. Model Architecture

| Layer | Details |
|-------|----------|
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | Sigmoid (5 labels) |

---

## 6. Example Output

```
Loaded train: 1000 images, 5 labels
Loaded valid: 200 images, 5 labels
Loaded test: 300 images, 5 labels
Test Accuracy: 92.15%
Model saved successfully as 'mehandi_multilabel_cnn.keras'
```

---

## 7. Dataset Information

- **Source:** [Roboflow – Mehandi Designs](https://universe.roboflow.com)  
- **License:** CC BY 4.0  
- **Preprocessing:** 640×640 image resize (stretch)  
- **Dataset Size:** ~1556 images  

---

## 8. Author

**Created by:** [@chillbasil](https://github.com/chillbasil)  

---

