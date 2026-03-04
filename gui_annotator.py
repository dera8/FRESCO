import sys
import json
import copy
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QListWidget, QLabel, QComboBox, 
                             QPushButton, QFormLayout, QMessageBox, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# Vocabularies
MATERIALS = ["asphalt", "concrete", "brick", "stone", "wood", "plaster", 
             "glass", "metal", "painted_surface", "tile", "slate", "shingles", "unknown"]
COLORS = ["black", "gray", "white", "red", "brown", "yellow", 
          "blue", "green", "beige", "orange", "unknown"]
MASK_COLORS = {
    "road": [255, 215, 0],
    "wall": [0, 255, 0],
    "roof": [0, 128, 255],
    "door": [0, 0, 255],
    "window": [255, 0, 128]
}

def normalize_path(path_str):
    """Convert Windows paths to Linux paths safely."""
    if not path_str:
        return None
    return str(path_str).replace('\\', '/')

class AnnotatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoPBR Material Annotator & GT Generator")
        self.setGeometry(100, 100, 1200, 800)

        self.manifest_data = []
        self.nemotron_data = {}
        self.current_image_data = None
        self.current_mask_id = None
        self.base_folder = None

        self.init_ui()
        
        # Automatically prompt for folder on startup
        self.open_folder_dialog()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL: Image List ---
        left_layout = QVBoxLayout()
        
        # 📂 NEW: Select Folder Button
        self.btn_folder = QPushButton("📂 Select Data Folder")
        self.btn_folder.clicked.connect(self.open_folder_dialog)
        self.btn_folder.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; font-weight: bold;")
        left_layout.addWidget(self.btn_folder)
        
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

        layout.addLayout(left_layout, 2)
        layout.addLayout(center_layout, 5)
        layout.addLayout(right_layout, 2)

    def open_folder_dialog(self):
        """Opens a dialog to select the folder containing the required JSON files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Output Folder (containing sam3_instances)")
        if folder:
            self.load_data_from_folder(folder)

    def load_data_from_folder(self, folder_path):
        """Validates and loads data from the selected folder."""
        folder = Path(folder_path)
        manifest_path = folder / "sam3_instances" / "manifest.json"
        nemotron_path = folder / "materials_full_filtered.json"
        
        if not manifest_path.exists() or not nemotron_path.exists():
            QMessageBox.warning(self, "Missing Files", 
                f"Could not find required files in:\n{folder}\n\n"
                f"Expected:\n- sam3_instances/manifest.json\n- materials_full_filtered.json")
            return
            
        self.base_folder = folder
        self.load_data(manifest_path, nemotron_path)

    def load_data(self, manifest_path, nemotron_path):
        try:
            # Clear previous data
            self.image_list.clear()
            self.mask_list.clear()
            self.manifest_data = []
            self.nemotron_data = {}
            self.current_image_data = None
            self.current_mask_id = None
            self.image_label.setText("Data loaded. Select an image.")

            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.manifest_data = json.load(f)
                
            with open(nemotron_path, 'r', encoding='utf-8') as f:
                full_nemotron = json.load(f)
                self.nemotron_meta = full_nemotron.get("meta", {})
                self.nemotron_data = full_nemotron.get("data", {})

            # Populate Image List
            for item in self.manifest_data:
                img_id = item.get("id", Path(normalize_path(item.get("rgb", item.get("image", "")))).stem)
                self.image_list.addItem(img_id)
            
            self.setWindowTitle(f"AutoPBR Material Annotator - {manifest_path.parent.parent.name}")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"Failed to load JSON files:\n{e}")

    def get_image_path(self):
        if not self.current_image_data: return None
        raw_path = self.current_image_data.get("rgb", self.current_image_data.get("image", ""))
        return normalize_path(raw_path)

    def match_manifest_instance(self, bid):
        """Smart matcher to connect building_05 to building_005"""
        instances = self.current_image_data.get("instances", [])
        
        # 1. Exact match
        for inst in instances:
            if inst.get("id") == bid:
                return inst
                
        # 2. Smart numeric match (fixes 2-digit to 3-digit transition)
        if bid.startswith("building_"):
            try:
                num = int(bid.split("_")[1])
                for inst in instances:
                    inst_id = inst.get("id", "")
                    if inst_id.startswith("building_") and int(inst_id.split("_")[1]) == num:
                        return inst
            except ValueError:
                pass
        return None

    def on_image_selected(self, item):
        img_id = item.text()
        self.current_image_data = next((img for img in self.manifest_data if img.get("id", Path(normalize_path(img.get("rgb", img.get("image", "")))).stem) == img_id), None)
        
        self.mask_list.clear()
        if not self.current_image_data:
            return

        self.mask_list.addItem("👁️ View All Global Masks")
        
        for cls in ["road", "wall", "roof", "door", "window"]:
            if f"{cls}_mask" in self.current_image_data.get("global", {}):
                self.mask_list.addItem(f"global_{cls}")
        
        predictions = self.nemotron_data.get(img_id, {})
        instances = predictions.get("instances", {})
        
        for bid, inst_data in sorted(instances.items()):
            if bid.startswith("building"):
                if "wall" in inst_data: self.mask_list.addItem(f"{bid}_wall")
                if "roof" in inst_data: self.mask_list.addItem(f"{bid}_roof")
                if "door" in inst_data: self.mask_list.addItem(f"{bid}_door")
            else:
                self.mask_list.addItem(bid)

        self.display_image(self.get_image_path())
        self.current_mask_id = None

    def on_mask_selected(self, item):
        mask_id = item.text()
        self.current_mask_id = mask_id
        
        img_id = self.current_image_data.get("id", Path(self.get_image_path()).stem)
        img_path = self.get_image_path()
        predictions = self.nemotron_data.get(img_id, {})
        
        # --- Handle "View All Global Masks" ---
        if mask_id == "👁️ View All Global Masks":
            masks_to_draw = []
            g_data = self.current_image_data.get("global", {})
            if "road_mask" in g_data: masks_to_draw.append((normalize_path(g_data["road_mask"]), MASK_COLORS["road"])) 
            if "wall_mask" in g_data: masks_to_draw.append((normalize_path(g_data["wall_mask"]), MASK_COLORS["wall"]))   
            if "roof_mask" in g_data: masks_to_draw.append((normalize_path(g_data["roof_mask"]), MASK_COLORS["roof"])) 
            if "door_mask" in g_data: masks_to_draw.append((normalize_path(g_data["door_mask"]), MASK_COLORS["door"])) 
            if "window_mask" in g_data: masks_to_draw.append((normalize_path(g_data["window_mask"]), MASK_COLORS["window"])) 
            
            self.display_multiple_overlays(img_path, masks_to_draw)
            self.combo_class.setEnabled(False)
            self.combo_color.setEnabled(False)
            return

        target_pred = None
        mask_path = None
        overlay_color = [255, 0, 0]
        
        if mask_id.startswith("global_"):
            cls = mask_id.split("_")[1]
            target_pred = predictions.get("global", {}).get(cls, {})
            mask_path = self.current_image_data.get("global", {}).get(f"{cls}_mask") or target_pred.get("mask")
            
            if cls == "road": overlay_color = MASK_COLORS["road"]
            elif cls == "wall": overlay_color = MASK_COLORS["wall"]
            elif cls == "roof": overlay_color = MASK_COLORS["roof"]
            elif cls == "door": overlay_color = MASK_COLORS["door"]
            elif cls == "window": overlay_color = MASK_COLORS["window"]
            
        else:
            if mask_id.endswith("_wall") or mask_id.endswith("_roof") or mask_id.endswith("_door") or mask_id.endswith("_window"):
                parts = mask_id.rsplit('_', 1) 
                bid, cls = parts[0], parts[1]
                target_pred = predictions.get("instances", {}).get(bid, {}).get(cls, {})
                
                # Use smart matcher to grab correct new path from manifest
                inst_manifest = self.match_manifest_instance(bid)
                
                if inst_manifest and inst_manifest.get(f"{cls}_mask"):
                    mask_path = inst_manifest.get(f"{cls}_mask")
                else:
                    mask_path = target_pred.get("mask")
                    if mask_path and bid.startswith("building_"):
                        num = int(bid.split("_")[1])
                        mask_path = mask_path.replace(f"building_{num:02d}", f"building_{num:03d}")
                            
                if cls == "wall": overlay_color = MASK_COLORS["wall"]
                elif cls == "roof": overlay_color = MASK_COLORS["roof"]
                elif cls == "door": overlay_color = MASK_COLORS["door"]
                elif cls == "window": overlay_color = MASK_COLORS["window"]
            else:
                bid = mask_id
                target_pred = predictions.get("instances", {}).get(bid, {})
                mask_path = target_pred.get("mask")
                if bid.startswith("road"):
                    overlay_color = [255, 0, 255]
                else:
                    overlay_color = [0, 255, 255]

        mask_path = normalize_path(mask_path)

        if mask_path and Path(mask_path).exists():
            self.display_multiple_overlays(img_path, [(mask_path, overlay_color)])
        else:
            self.display_image(img_path)

        # Editor Controls
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
        if not self.current_image_data or not self.current_mask_id:
            return
            
        img_id = self.current_image_data.get("id", Path(self.get_image_path()).stem)
        
        if self.current_mask_id.startswith("global_"):
            cls = self.current_mask_id.split("_")[1]
            target = self.nemotron_data.get(img_id, {}).get("global", {}).get(cls, {})
        else:
            if self.current_mask_id.endswith("_wall") or self.current_mask_id.endswith("_roof") or self.current_mask_id.endswith("_door") or self.current_mask_id.endswith("_window"):
                parts = self.current_mask_id.rsplit('_', 1) 
                bid, cls = parts[0], parts[1]
                target = self.nemotron_data.get(img_id, {}).get("instances", {}).get(bid, {}).get(cls, {})
            else:
                bid = self.current_mask_id
                target = self.nemotron_data.get(img_id, {}).get("instances", {}).get(bid, {})

        if target and not target.get("skipped", True):
            if "material_descriptor" not in target:
                target["material_descriptor"] = {}
                
            target["material_descriptor"]["class"] = self.combo_class.currentText()
            target["material_descriptor"]["color"] = self.combo_color.currentText()
            target["material_descriptor"]["confidence"] = 1.0  

    def display_image(self, img_path):
        if not img_path or not Path(img_path).exists():
            self.image_label.setText(f"Image not found:\n{img_path}")
            return
        pixmap = QPixmap(img_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_multiple_overlays(self, img_path, masks_data):
        if not img_path or not Path(img_path).exists():
            self.image_label.setText(f"Image not found:\n{img_path}")
            return
            
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        alpha = 0.5
        
        for mask_path, color_rgb in masks_data:
            if not mask_path or not Path(mask_path).exists():
                continue
                
            mask_pil = Image.open(mask_path).convert("L")
            if mask_pil.size != img_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
                
            mask_np = np.array(mask_pil)
            mask_indices = mask_np > 127
            
            colored_mask = np.zeros_like(img_np)
            colored_mask[:, :] = color_rgb
            
            img_np[mask_indices] = (img_np[mask_indices] * (1 - alpha) + colored_mask[mask_indices] * alpha).astype(np.uint8)
        
        h, w, ch = img_np.shape
        bytes_per_line = ch * w
        q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_ground_truth(self):
        output_data = {
            "meta": self.nemotron_meta,
            "data": {}
        }
        
        output_data["meta"]["version"] = "v3.0_merged_HUMAN_GT"
        output_data["meta"]["note"] = "Human-verified Ground Truth (Global sections removed)"
        
        for img_id, img_data in self.nemotron_data.items():
            clean_img_data = copy.deepcopy(img_data)
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