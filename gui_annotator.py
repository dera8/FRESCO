import sys
import json
import copy
import numpy as np
from pathlib import Path
from PIL import Image  # <-- Added this
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QListWidget, QLabel, QComboBox, 
                             QPushButton, QFormLayout, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# Vocabularies from your Nemotron script
MATERIALS = ["asphalt", "concrete", "brick", "stone", "wood", "plaster", 
             "glass", "metal", "painted_surface", "tile", "slate", "shingles", "unknown"]
COLORS = ["black", "gray", "white", "red", "brown", "yellow", 
          "blue", "green", "beige", "orange", "unknown"]

class AnnotatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoPBR Material Annotator & GT Generator")
        self.setGeometry(100, 100, 1200, 800)

        # In-memory data
        self.manifest_data = []
        self.nemotron_data = {}
        self.current_image_data = None
        self.current_mask_id = None

        self.init_ui()
        
        # NOTE: Update these paths if your files are located somewhere else!
        self.load_data(
            manifest_path="data_output/sam3_instances/manifest.json", 
            nemotron_path="materials_v2_filtered.json"
        )

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL: Image List ---
        left_layout = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        left_layout.addWidget(QLabel("1. Select Image:"))
        left_layout.addWidget(self.image_list)
        
        # --- CENTER PANEL: Image Viewer ---
        center_layout = QVBoxLayout()
        self.image_label = QLabel("Select an image to view...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; color: white;")
        center_layout.addWidget(self.image_label)

        # --- RIGHT PANEL: Mask List & Editor ---
        right_layout = QVBoxLayout()
        
        self.mask_list = QListWidget()
        self.mask_list.itemClicked.connect(self.on_mask_selected)
        right_layout.addWidget(QLabel("2. Select Region:"))
        right_layout.addWidget(self.mask_list)

        # Editor Form
        form_layout = QFormLayout()
        self.combo_class = QComboBox()
        self.combo_class.addItems(MATERIALS)
        self.combo_color = QComboBox()
        self.combo_color.addItems(COLORS)
        
        # Connect comboboxes to the save function so they update memory in real-time
        self.combo_class.currentTextChanged.connect(self.update_annotation)
        self.combo_color.currentTextChanged.connect(self.update_annotation)
        
        form_layout.addRow("Material:", self.combo_class)
        form_layout.addRow("Color:", self.combo_color)
        
        right_layout.addLayout(form_layout)

        # Save Button
        self.btn_save = QPushButton("💾 Save Image Annotations to GT")
        self.btn_save.clicked.connect(self.save_ground_truth)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        right_layout.addWidget(self.btn_save)

        # Assemble Layout
        layout.addLayout(left_layout, 2)
        layout.addLayout(center_layout, 5)
        layout.addLayout(right_layout, 2)

    def load_data(self, manifest_path, nemotron_path):
        """Load and merge your SAM3 manifest and Nemotron predictions."""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.manifest_data = json.load(f)
                
            with open(nemotron_path, 'r', encoding='utf-8') as f:
                full_nemotron = json.load(f)
                # Keep the meta section for when we save later
                self.nemotron_meta = full_nemotron.get("meta", {})
                self.nemotron_data = full_nemotron.get("data", {})

            for item in self.manifest_data:
                img_id = item.get("id", Path(item.get("image", "")).stem)
                self.image_list.addItem(img_id)
                
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"Failed to load JSON files:\n{e}")

    def on_image_selected(self, item):
        """Triggered when user clicks an image in the left list."""
        img_id = item.text()
        
        # Find image in manifest
        self.current_image_data = next((img for img in self.manifest_data if img.get("id", Path(img.get("image", "")).stem) == img_id), None)
        
        # Populate Mask List (Right Panel)
        self.mask_list.clear()
        if self.current_image_data:
            # Add a composite view option at the very top
            self.mask_list.addItem("👁️ View All Global Masks")
            
            # Add individual globals
            for cls in ["road", "wall", "roof"]:
                if f"{cls}_mask" in self.current_image_data.get("global", {}):
                    self.mask_list.addItem(f"global_{cls}")
            
            # Add instances
            for inst in self.current_image_data.get("instances", []):
                bid = inst.get("id")
                self.mask_list.addItem(f"{bid}_wall")
                self.mask_list.addItem(f"{bid}_roof")

        # Show raw image initially
        self.display_image(self.current_image_data["image"])
        self.current_mask_id = None

    def on_mask_selected(self, item):
        """Triggered when user clicks a specific mask."""
        mask_id = item.text()
        self.current_mask_id = mask_id
        
        img_id = self.current_image_data.get("id", Path(self.current_image_data["image"]).stem)
        img_path = self.current_image_data["image"]
        
        # --- Handle "View All Global Masks" ---
        if mask_id == "👁️ View All Global Masks":
            masks_to_draw = []
            g_data = self.current_image_data.get("global", {})
            if "road_mask" in g_data: masks_to_draw.append((g_data["road_mask"], [255, 215, 0])) # Yellow
            if "wall_mask" in g_data: masks_to_draw.append((g_data["wall_mask"], [0, 255, 0]))   # Green
            if "roof_mask" in g_data: masks_to_draw.append((g_data["roof_mask"], [0, 128, 255])) # Blue
            
            self.display_multiple_overlays(img_path, masks_to_draw)
            
            # Disable editor (can't edit all 3 at once)
            self.combo_class.setEnabled(False)
            self.combo_color.setEnabled(False)
            return

        # --- Handle Individual Masks ---
        predictions = self.nemotron_data.get(img_id, {})
        target_pred = None
        mask_path = None
        overlay_color = [255, 0, 0] # Default Red
        
        if mask_id.startswith("global_"):
            cls = mask_id.split("_")[1]
            mask_path = self.current_image_data.get("global", {}).get(f"{cls}_mask")
            target_pred = predictions.get("global", {}).get(cls, {})
            # Match colors
            if cls == "road": overlay_color = [255, 215, 0]
            elif cls == "wall": overlay_color = [0, 255, 0]
            elif cls == "roof": overlay_color = [0, 128, 255]
            
        else:
            parts = mask_id.rsplit('_', 1) 
            bid, cls = parts[0], parts[1]
            inst = next((i for i in self.current_image_data.get("instances", []) if i.get("id") == bid), None)
            mask_path = inst.get(f"{cls}_mask") if inst else None
            target_pred = predictions.get("instances", {}).get(bid, {}).get(cls, {})
            # Match colors for instances too
            if cls == "wall": overlay_color = [0, 255, 0]
            elif cls == "roof": overlay_color = [0, 128, 255]

        # Render Overlay
        if mask_path and Path(mask_path).exists():
            self.display_multiple_overlays(img_path, [(mask_path, overlay_color)])
        else:
            self.display_image(img_path)

        # Populate Editor Comboboxes
        self.combo_class.blockSignals(True)
        self.combo_color.blockSignals(True)
        
        if target_pred and not target_pred.get("skipped", True):
            mat_desc = target_pred.get("material_descriptor", {})
            class_val = mat_desc.get("class", "unknown")
            color_val = mat_desc.get("color", "unknown")
            
            if class_val not in [self.combo_class.itemText(i) for i in range(self.combo_class.count())]:
                self.combo_class.addItem(class_val)
            if color_val not in [self.combo_color.itemText(i) for i in range(self.combo_color.count())]:
                self.combo_color.addItem(color_val)
                
            self.combo_class.setCurrentText(class_val)
            self.combo_color.setCurrentText(color_val)
            self.combo_class.setEnabled(True)
            self.combo_color.setEnabled(True)
        else:
            self.combo_class.setCurrentText("unknown")
            self.combo_color.setCurrentText("unknown")
            self.combo_class.setEnabled(False)
            self.combo_color.setEnabled(False)
            
        self.combo_class.blockSignals(False)
        self.combo_color.blockSignals(False)

    def update_annotation(self):
        """Saves combobox changes back to the in-memory nemotron dictionary."""
        if not self.current_image_data or not self.current_mask_id:
            return
            
        img_id = self.current_image_data.get("id", Path(self.current_image_data["image"]).stem)
        
        # Navigate to the correct prediction in our dictionary
        if self.current_mask_id.startswith("global_"):
            cls = self.current_mask_id.split("_")[1]
            target = self.nemotron_data.get(img_id, {}).get("global", {}).get(cls, {})
        else:
            parts = self.current_mask_id.rsplit('_', 1) 
            bid, cls = parts[0], parts[1]
            target = self.nemotron_data.get(img_id, {}).get("instances", {}).get(bid, {}).get(cls, {})

        # If it's a valid mask, update the material_descriptor
        if target and not target.get("skipped", True):
            if "material_descriptor" not in target:
                target["material_descriptor"] = {}
                
            target["material_descriptor"]["class"] = self.combo_class.currentText()
            target["material_descriptor"]["color"] = self.combo_color.currentText()
            target["material_descriptor"]["confidence"] = 1.0  # Force confidence to 1.0 because a human verified it

    def display_image(self, img_path):
        """Load and scale RGB image to center panel."""
        if not Path(img_path).exists():
            self.image_label.setText("Image not found on disk.")
            return
        pixmap = QPixmap(img_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_multiple_overlays(self, img_path, masks_data):
        """
        Blend multiple binary masks over the RGB image.
        masks_data is a list of tuples: [(mask_path, [R, G, B]), ...]
        """
        if not Path(img_path).exists():
            self.image_label.setText("Image not found on disk.")
            return
            
        # Load base image
        from PIL import Image
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        
        alpha = 0.5
        
        for mask_path, color_rgb in masks_data:
            if not Path(mask_path).exists():
                continue
                
            mask_pil = Image.open(mask_path).convert("L")
            if mask_pil.size != img_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
                
            mask_np = np.array(mask_pil)
            mask_indices = mask_np > 127
            
            # Create a color layer for this specific mask
            colored_mask = np.zeros_like(img_np)
            colored_mask[:, :] = color_rgb
            
            # Blend it
            img_np[mask_indices] = (img_np[mask_indices] * (1 - alpha) + colored_mask[mask_indices] * alpha).astype(np.uint8)
        
        # Convert back to PyQt format
        h, w, ch = img_np.shape
        bytes_per_line = ch * w
        q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        
    def save_ground_truth(self):
        """Save annotations to JSON, removing the 'global' sections."""
        # Build the final output structure
        output_data = {
            "meta": self.nemotron_meta,
            "data": {}
        }
        
        # Add a note that this is Human GT
        output_data["meta"]["version"] = "v2.1_filtered_HUMAN_GT"
        output_data["meta"]["note"] = "Human-verified Ground Truth (Global sections removed)"
        
        for img_id, img_data in self.nemotron_data.items():
            # Deep copy so we don't delete 'global' from the active GUI session
            clean_img_data = copy.deepcopy(img_data)
            
            # Remove the 'global' dictionary as requested
            if "global" in clean_img_data:
                del clean_img_data["global"]
                
            output_data["data"][img_id] = clean_img_data
            
        out_path = Path("ground_truth_eval.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Saved successfully to:\n{out_path.absolute()}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save Ground Truth:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnnotatorGUI()
    ex.show()
    sys.exit(app.exec_())