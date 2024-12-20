import logging
import os
from pathlib import Path
import re

import slicer
from slicer.ScriptedLoadableModule import *
import ctk
import qt
import SimpleITK as sitk
import sitkUtils
import numpy as np
from typing import Iterable, Optional, Union

logging.basicConfig(level=logging.INFO)

#
# NIfTILazyDataLoader
#


class NIfTILazyDataLoader(ScriptedLoadableModule):
    def __init__(self, parent):
        """
        Initialize the MRIAnnotator module.

        Parameters:
        - parent: The parent module.
        """
        parent.title = "NIfTI Lazy DataLoader"
        parent.categories = ["Machine Learning"]
        parent.contributors = ["Alejandro Mora-Rubio ()"]
        self.parent = parent


#
# NIfTILazyDataLoaderWidget
#


class NIfTILazyDataLoaderWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        """
        Set up the UI elements and initialize the widget.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.layout.setContentsMargins(10, 10, 10, 10)  # Left, Top, Right, Bottom

        # Collapsible Configuration Menu
        self.configMenuButton = ctk.ctkCollapsibleButton()
        self.configMenuButton.text = "Configuration Menu"
        self.configMenuButton.collapsed = True  # Start collapsed
        self.layout.addWidget(self.configMenuButton)
        configMenuLayout = qt.QVBoxLayout(self.configMenuButton)

        # ComboBox for Configuration Options
        self.configComboBox = qt.QComboBox()
        self.configComboBox.addItems(["nnUNet dataset", "Patient/Images & Labels", "2 independent directories"])
        self.configComboBox.setToolTip("Select the directory type.")
        self.configComboBox.currentIndexChanged.connect(self.onComboBoxChanged)
        configMenuLayout.addWidget(qt.QLabel("Directory Type:"))
        configMenuLayout.addWidget(self.configComboBox)
        # Create a group box to hold the radio buttons
        self.modeGroupBox = qt.QGroupBox("Mode")
        self.modeGroupBox.setEnabled(self.configComboBox.currentText == "nnUNet dataset")
        modeLayout = qt.QHBoxLayout(self.modeGroupBox)
        self.trainRadioButton = qt.QRadioButton("Train")
        testRadioButton = qt.QRadioButton("Test")
        # Set default selection
        self.trainRadioButton.setChecked(True)
        # Add the radio buttons to their horizontal layout
        modeLayout.addWidget(self.trainRadioButton)
        modeLayout.addWidget(testRadioButton)
        # Add the group box to the main configuration layout
        configMenuLayout.addWidget(self.modeGroupBox)
        # --- Input Text Fields ---
        # Create a form layout for the labeled line edits
        regexFormLayout = qt.QFormLayout()
        self.imagesRegexLineEdit = qt.QLineEdit()
        self.imagesRegexLineEdit.text = "^\w+_(?P<modality>(?!seg\.nii\.gz$)\w{1,5})\.nii\.gz$"
        self.labelsRegexLineEdit = qt.QLineEdit()
        self.labelsRegexLineEdit.text = "^\w+_seg\.nii\.gz$"
        regexFormLayout.addRow("Images regex:", self.imagesRegexLineEdit)
        regexFormLayout.addRow("Labels regex:", self.labelsRegexLineEdit)
        self.imagesRegexLineEdit.setEnabled(self.configComboBox.currentText != "nnUNet dataset")
        self.labelsRegexLineEdit.setEnabled(self.configComboBox.currentText != "nnUNet dataset")
        # Add the form layout to the main configuration layout
        configMenuLayout.addLayout(regexFormLayout)
        self.layout.addLayout(configMenuLayout)

        # Horizontal layout for label and directory input
        imageDirectoryInputLayout = qt.QHBoxLayout()
        # Label widget
        imageDirectoryLabel = qt.QLabel("Images Directory:")
        imageDirectoryLabel.setToolTip("Select a directory containing the files.")
        imageDirectoryInputLayout.addWidget(imageDirectoryLabel)
        # Directory path input
        self.imageDirectoryPathEdit = ctk.ctkPathLineEdit()
        self.imageDirectoryPathEdit.filters = ctk.ctkPathLineEdit.Dirs  # Allow only directories
        self.imageDirectoryPathEdit.setToolTip("Select a directory containing the files.")
        self.imageDirectoryPathEdit.showHistoryButton = False  # Enable history button
        self.imageDirectoryPathEdit.currentPathChanged.connect(self.onDirectoryPathEditSelected)
        imageDirectoryInputLayout.addWidget(self.imageDirectoryPathEdit)
        self.layout.addLayout(imageDirectoryInputLayout)

        # Horizontal layout for label and directory input
        labelDirectoryInputLayout = qt.QHBoxLayout()
        # Label widget
        labelDirectoryLabel = qt.QLabel("Labels Directory:")
        labelDirectoryLabel.setToolTip("Select a directory containing the files.")
        labelDirectoryInputLayout.addWidget(labelDirectoryLabel)
        # Directory path input
        self.labelDirectoryPathEdit = ctk.ctkPathLineEdit()
        self.labelDirectoryPathEdit.filters = ctk.ctkPathLineEdit.Dirs  # Allow only directories
        self.labelDirectoryPathEdit.setToolTip("Select a directory containing the labels.")
        self.labelDirectoryPathEdit.showHistoryButton = False  # Enable history button
        self.labelDirectoryPathEdit.currentPathChanged.connect(self.onDirectoryPathEditSelected)
        self.labelDirectoryPathEdit.setEnabled(self.configComboBox.currentText == "2 independent directories")
        labelDirectoryInputLayout.addWidget(self.labelDirectoryPathEdit)
        self.layout.addLayout(labelDirectoryInputLayout)
        
        # Button to search directory
        self.searchDirectoryButton = qt.QPushButton("Search Directory")
        self.layout.addWidget(self.searchDirectoryButton)
        self.searchDirectoryButton.connect('clicked(bool)', self.search_directory)

        # File List View
        self.layout.addWidget(qt.QLabel("Available cases:"))
        self.fileListWidget = qt.QListWidget()
        self.fileListWidget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.layout.addWidget(self.fileListWidget)

        # Button to load
        self.loadButton = qt.QPushButton("Load")
        self.layout.addWidget(self.loadButton)
        self.loadButton.connect('clicked(bool)', self.onLoadButton)

        # Horizontal layout for previous and next buttons
        previousNextButtonLayout = qt.QHBoxLayout()
        # Button to load previous images
        self.previousButton = qt.QPushButton("Previous")
        self.previousButton.connect('clicked(bool)', self.onPreviousButton)
        previousNextButtonLayout.addWidget(self.previousButton)
        # Button to load next images
        self.nextButton = qt.QPushButton("Next")
        self.nextButton.connect('clicked(bool)', self.onNextButton)
        previousNextButtonLayout.addWidget(self.nextButton)
        self.layout.addLayout(previousNextButtonLayout)

        self.layout.addWidget(qt.QLabel("Current case:"))
        self.sceneTreeView = slicer.qMRMLSubjectHierarchyTreeView()
        self.sceneTreeView.setMRMLScene(slicer.mrmlScene)  # Tree view for displaying loaded nodes
        self.layout.addWidget(self.sceneTreeView)

        self.availableCases = {}

    def search_directory(self):
        """
        Populate the file list widget with filenames from the selected directory.
        """
        logging.debug("Searching directory...")
        self.availableCases = {}
        self.fileListWidget.clear()
        if self.configComboBox.currentText == "nnUNet dataset":
            if self.imageDirectoryPathEdit.currentPath == "":
                return
            self.navigate_folder_nnunet()
        elif self.configComboBox.currentText == "Patient/Images & Labels":
            if self.imageDirectoryPathEdit.currentPath == "":
                return
            self.navigate_folder_patient()
        elif self.configComboBox.currentText == "2 independent directories":
            if (self.imageDirectoryPathEdit.currentPath == "") or (self.labelDirectoryPathEdit.currentPath == ""):
                return
            self.navigate_folder_two_directories()

    def onDirectoryPathEditSelected(self, path):
        self.search_directory()

    def navigate_folder_nnunet(self):
        """
        Navigate provided images and labels directories.
        """
        try:
            suffix = "Tr" if self.trainRadioButton.isChecked() else "Ts" 
            images_path = Path(self.imageDirectoryPathEdit.currentPath) / f"images{suffix}"
            labels_path = Path(self.imageDirectoryPathEdit.currentPath) / f"labels{suffix}"
            cases = []
            for label in labels_path.iterdir():
                case_id = label.name.split('.')[0]
                logging.debug(f"Looking for images for Case ID: {case_id}")
                images = list(images_path.glob(f"{case_id}*"))
                if images:
                    logging.debug(f"Found {len(images)} images for Case ID: {case_id}")
                    cases.append(case_id)
                    self.availableCases[case_id] = {
                        "images": images,
                        "label": label,
                    }
                else:
                    logging.debug(f"No images for Case ID: {case_id}")
            self.fileListWidget.addItems(sorted(cases))
            logging.info(f"Added {len(cases)} cases.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to navigate input directory: {str(e)}")
    
    def navigate_folder_patient(self):
        """
        Navigate provided directory in Patient/Images & Labels structure.
        """
        if (not self.imagesRegexLineEdit.text) or (not self.labelsRegexLineEdit.text):
            slicer.util.errorDisplay("Please provide regex for images and labels.")
            logging.error("Please provide regex for images and labels.")
            return
        try:
            dir_path = Path(self.imageDirectoryPathEdit.currentPath)
            cases = []
            for patient in dir_path.iterdir():
                if (not patient.is_dir()) or patient.name.startswith("."):
                    continue
                case_id = patient.name
                logging.debug(f"Looking for images for Case ID: {case_id}")
                files = list(patient.glob("*"))
                images = []
                label = None
                for file in files:
                    if file.name.startswith("."):
                        continue
                    if re.match(self.imagesRegexLineEdit.text, file.name):
                        images.append(file)
                    if re.match(self.labelsRegexLineEdit.text, file.name):
                        label = file
                if images:
                    logging.debug(f"Found {len(images)} images for Case ID: {case_id}")
                    cases.append(case_id)
                    self.availableCases[case_id] = {
                        "images": images,
                        "label": label,
                    }
                else:
                    logging.debug(f"No images found for Case ID: {case_id}")
                if not label:
                    logging.warning(f"No label found for Case ID: {case_id}")
            self.fileListWidget.addItems(sorted(cases))
            logging.info(f"Added {len(cases)} cases.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to navigate input directory: {str(e)}")

    def navigate_folder_two_directories(self):
        """
        Navigate provided directories. Filenames for images and labels must match.
        """
        try:
            images_path = Path(self.imageDirectoryPathEdit.currentPath)
            labels_path = Path(self.labelDirectoryPathEdit.currentPath)
            cases = []
            for label in labels_path.iterdir():
                if label.name.startswith("."):
                    continue
                case_id = label.name.split(".")[0]
                logging.debug(f"Looking for image for Case ID: {case_id}")
                image = images_path / label.name
                if image.exists():
                    logging.debug(f"Found image for Case ID: {case_id}")
                    cases.append(case_id)
                    self.availableCases[case_id] = {
                        "images": [image],
                        "label": label,
                    }
                else:
                    logging.debug(f"No images for Case ID: {case_id}")
            self.fileListWidget.addItems(sorted(cases))
            logging.info(f"Added {len(cases)} cases.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to navigate input directory: {str(e)}")

    def load_selected_case(self):
        """
        Load the selected files into 3D Slicer.
        """
        slicer.mrmlScene.Clear(0)
        selected_item = self.fileListWidget.selectedItems()[0]
        case_id = selected_item.text()
        logging.debug(f"Selected case: {case_id}")
        images = self.availableCases[case_id]["images"]
        label = self.availableCases[case_id]["label"]
        for image in images:
            try:
                img_node = slicer.util.loadVolume(image)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load image {image}: {str(e)}")
        if label:
            try:
                seg_node = slicer.util.loadSegmentation(label)
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load label {label}: {str(e)}")
        else:
            logging.info(f"No label for Case ID: {case_id}")

    def onLoadButton(self):
        """
        Load the selected files into 3D Slicer.
        """
        self.load_selected_case()

    def onNextButton(self):
        """Move to the next item in the file list."""
        current_row = self.fileListWidget.currentRow
        if current_row < self.fileListWidget.count - 1:  # Ensure not at the last item
            self.fileListWidget.setCurrentRow(current_row + 1)
        self.load_selected_case()


    def onPreviousButton(self):
        """Move to the previous item in the file list."""
        current_row = self.fileListWidget.currentRow
        if current_row > 0:  # Ensure not at the last item
            self.fileListWidget.setCurrentRow(current_row - 1)
        self.load_selected_case()

    def onComboBoxChanged(self):
        slicer.mrmlScene.Clear(0)
        self.availableCases = {}
        self.fileListWidget.clear()

        self.modeGroupBox.setEnabled(self.configComboBox.currentText == "nnUNet dataset")
        self.labelDirectoryPathEdit.setEnabled(self.configComboBox.currentText == "2 independent directories")
        self.imagesRegexLineEdit.setEnabled(self.configComboBox.currentText == "Patient/Images & Labels")
        self.labelsRegexLineEdit.setEnabled(self.configComboBox.currentText == "Patient/Images & Labels")