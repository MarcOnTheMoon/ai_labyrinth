"""
Main Application and GUI to parametrize the training or the evaluation

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QLineEdit, QSpinBox, QFormLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt

from LabApplicationFunction import AppFunction

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
pictures_dir = os.path.join(current_dir, "pictures")

class ApplicationGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.setWindowTitle("Labyrinth GUI")
        self.setGeometry(100, 100, 600, 400)  # x, y, width, height
        self.setWindowIcon(QIcon(os.path.join(pictures_dir, "HAW.png")))

        main_layout = QVBoxLayout()  # Main layout as vertical layout

        top_layout = QHBoxLayout() # Top layout for the two halves (left and right side)

        # ========== left layout ==================================================
        left_layout = QVBoxLayout() # Define left layout

        # Checkboxes for environment
        env_group = QGroupBox("Which environment should be used?")
        env_layout = QHBoxLayout()

        self.__virt_cb = QCheckBox("Virtual")
        self.__virt_cb.setChecked(True)  # Initially select "Virtual"
        self.__proto_cb = QCheckBox("Prototype")
        env_layout.addWidget(self.__virt_cb)
        env_layout.addWidget(self.__proto_cb)

        env_group.setLayout(env_layout)
        left_layout.addWidget(env_group)

        # Checkboxes for training or evaluation
        mode_group = QGroupBox("Would you like to train or evaluate?")
        mode_layout = QHBoxLayout()

        self.__train_cb = QCheckBox("Train")
        self.__eval_cb = QCheckBox("Evaluate")
        self.__eval_cb.setChecked(True)  # Initially select "Evaluated"
        mode_layout.addWidget(self.__train_cb)
        mode_layout.addWidget(self.__eval_cb)

        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)

        # Additional option for further training (dynamically visible)
        self.__continue_training_group = QGroupBox("Should training continue?")
        continue_training_layout = QHBoxLayout()

        self.__continue_train_cb = QCheckBox("Yes")
        self.__no_continue_train_cb = QCheckBox("No")
        self.__no_continue_train_cb.setChecked(True)
        continue_training_layout.addWidget(self.__continue_train_cb)
        continue_training_layout.addWidget(self.__no_continue_train_cb)

        self.__continue_training_group.setLayout(continue_training_layout)
        self.__continue_training_group.setVisible(False)
        left_layout.addWidget(self.__continue_training_group)

        # Combobox for game board
        plate_group = QGroupBox("Which game board would you like to play?")
        plate_layout = QHBoxLayout()

        self.__plate_cb = QComboBox()
        self.__plate_cb.addItems(["HOLES_0_VIRTUAL", "HOLES_0", "HOLES_2_VIRTUAL", "HOLES_2", "HOLES_8", "HOLES_21"])
        plate_layout.addWidget(self.__plate_cb)

        plate_group.setLayout(plate_layout)
        left_layout.addWidget(plate_group)

        # Image of the labyrinth (bottom left)
        self.__labyrinth_image_label = QLabel()
        self.__labyrinth_image_label.setFixedSize(250, 250)  # Limit image size
        # A layout to center the image
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.__labyrinth_image_label)
        image_layout.addStretch()

        left_layout.addLayout(image_layout)
        top_layout.addLayout(left_layout)  # Add left layout to top layout

        self.__update_labyrinth_image()  # Show image directly at startup

        # ========== right layout =================================================
        # Right layout for training parameters and network structure
        right_layout = QVBoxLayout()

        # Network layers
        net_group = QGroupBox("Network Structure")
        net_layout = QFormLayout()

        self.__num_layers_input = QSpinBox()
        self.__num_layers_input.setRange(1, 3)
        self.__num_layers_input.setValue(3)

        # Labels for the number of neurons
        self.__layer1_label = QLabel("Neurons in Layer 1:")
        self.__layer2_label = QLabel("Neurons in Layer 2:")
        self.__layer3_label = QLabel("Neurons in Layer 3:")

        self.__neurons_layer1_input = QLineEdit("64")
        self.__neurons_layer2_input = QLineEdit("64")
        self.__neurons_layer3_input = QLineEdit("64")
        self.__neurons_layer2_input.setEnabled(False)
        self.__neurons_layer3_input.setEnabled(False)

        net_layout.addRow("Number of Layers:", self.__num_layers_input)
        net_layout.addRow(self.__layer1_label, self.__neurons_layer1_input)
        net_layout.addRow(self.__layer2_label, self.__neurons_layer2_input)
        net_layout.addRow(self.__layer3_label, self.__neurons_layer3_input)

        net_group.setLayout(net_layout)
        right_layout.addWidget(net_group)

        # Parameter inputs
        self.__param_group = QGroupBox("Training Parameters")
        param_layout = QFormLayout()

        self.__episodes_input = QLineEdit("1000")
        self.__seed_input = QLineEdit("1")
        self.__epsilon_input = QLineEdit("1.0")
        self.__epsilon_decay_rate_input = QLineEdit("0.98")
        self.__epsilon_min_input = QLineEdit("0.15")
        self.__batch_size_input = QLineEdit("64")
        self.__replay_buffer_size_input = QLineEdit("100000")
        self.__gamma_input = QLineEdit("0.99")
        self.__learning_rate_input = QLineEdit("5e-4")
        self.__learn_period_input = QLineEdit("1")

        param_layout.addRow("Episodes:", self.__episodes_input)
        param_layout.addRow("Seed:", self.__seed_input)
        param_layout.addRow("Epsilon:", self.__epsilon_input)
        param_layout.addRow("Epsilon Decay Rate:", self.__epsilon_decay_rate_input)
        param_layout.addRow("Epsilon Min:", self.__epsilon_min_input)
        param_layout.addRow("Batch Size:", self.__batch_size_input)
        param_layout.addRow("Replay Buffer Size:", self.__replay_buffer_size_input)
        param_layout.addRow("Gamma:", self.__gamma_input)
        param_layout.addRow("Learning Rate:", self.__learning_rate_input)
        param_layout.addRow("Learn Period:", self.__learn_period_input)

        self.__param_group.setLayout(param_layout)
        right_layout.addWidget(self.__param_group)
        self.__param_group.setVisible(False) # initial not visible

        self.__update_parameter_neurons()

        # Add spacer to fill space between network group and start button
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Add start button and error message label to the right layout
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.__start_game)
        start_button.setStyleSheet(""" QPushButton {background-color: rgb(68,74,250); color: white;}""")
        right_layout.addWidget(start_button)

        #Error message if some inputs are wrong
        self.__error_label = QLabel()
        self.__error_label.setStyleSheet("color: red;")  # Set text color to red
        self.__error_label.setVisible(False)  # Label hidden by default
        right_layout.addWidget(self.__error_label)

        top_layout.addLayout(right_layout)  # Add right layout to top layout

        main_layout.addLayout(top_layout)  # Add top layout to main layout

        self.setLayout(main_layout)

        # ========== connect signals ==============================================
        self.__virt_cb.stateChanged.connect(self.__toggle_environment)
        self.__proto_cb.stateChanged.connect(self.__toggle_environment)

        self.__train_cb.stateChanged.connect(self.__toggle_mode)
        self.__eval_cb.stateChanged.connect(self.__toggle_mode)

        self.__continue_train_cb.stateChanged.connect(self.__toggle_continue_train)
        self.__no_continue_train_cb.stateChanged.connect(self.__toggle_continue_train)

        self.__plate_cb.currentIndexChanged.connect(self.__update_labyrinth_image)
        self.__plate_cb.currentIndexChanged.connect(self.__update_parameter_neurons)

        self.__num_layers_input.valueChanged.connect(self.__update_neurons_inputs)


    def __toggle_environment(self, state):
        """
            toggles the checkboxes for the Environment (virtual or prototype)

            Parameters
            ----------
            state: Qt.CheckState

            Returns
            -------
            None.

        """
        if self.sender() == self.__virt_cb and state == Qt.CheckState.Checked.value:
            self.__proto_cb.setChecked(False)
        elif self.sender() == self.__proto_cb and state == Qt.CheckState.Checked.value:
            self.__virt_cb.setChecked(False)

    def __toggle_mode(self, state):
        """
            toggles the checkboxes for the modes (train or evaluate)

            Parameters
            ----------
            state: Qt.CheckState

            Returns
            -------
            None.

        """
        if self.sender() == self.__train_cb and state == Qt.CheckState.Checked.value:
            self.__eval_cb.setChecked(False)  # Uncheck eval_cb
            self.__continue_training_group.setVisible(True)
            self.__param_group.setVisible(True)  # Show the training parameters
        elif self.sender() == self.__eval_cb and state == Qt.CheckState.Checked.value:
            self.__train_cb.setChecked(False)  # Uncheck train_cb
            self.__continue_training_group.setVisible(False)
            self.__param_group.setVisible(False)

    def __toggle_continue_train(self, state):
        """
            toggles the checkboxes for the continue training (training with already pre-trained agents or not)

            Parameters
            ----------
            state: Qt.CheckState

            Returns
            -------
            None.

        """
        if self.sender() == self.__continue_train_cb and state == Qt.CheckState.Checked.value:
            self.__no_continue_train_cb.setChecked(False)
        elif self.sender() == self.__no_continue_train_cb and state == Qt.CheckState.Checked.value:
            self.__continue_train_cb.setChecked(False)

    def __update_labyrinth_image(self):
        """
            Load and scale image based on selection in combobox

            Parameters
            ----------
            None

            Returns
            -------
            None.

        """
        selected_plate = self.__plate_cb.currentText()
        pixmap_path = os.path.join(pictures_dir, f"{selected_plate}.png")
        pixmap = QPixmap(pixmap_path)
        scaled_pixmap = pixmap.scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.__labyrinth_image_label.setPixmap(scaled_pixmap)

    def __update_parameter_neurons(self):
        """
            updates all parameters and neurons to standard initializations

            Parameters
            ----------
            None

            Returns
            -------
            None.

        """
        selected_plate = self.__plate_cb.currentText()
        # Define parameters for each game board
        parameters = {
            "HOLES_0_VIRTUAL": {"layers": [512, 128], "epsilon_decay_rate": "0.9", "epsilon_min": "0.1"},
            "HOLES_0": {"layers": [512, 128], "epsilon_decay_rate": "0.9", "epsilon_min": "0.1"},
            "HOLES_2_VIRTUAL": {"layers": [128, 128]},
            "HOLES_2": {"layers": [128, 128]},
            "HOLES_8": {"layers": [2048, 1024, 256], "episodes": "3500"},
            "HOLES_21": {"layers": [2048, 1024, 256], "episodes": "3500"}
        }

        if selected_plate in parameters:
            params = parameters[selected_plate]

            episodes = params.get("episodes", "1000")
            seed = params.get("seed", "1")
            epsilon = params.get("epsilon", "1.0")
            epsilon_decay_rate = params.get("epsilon_decay_rate", "0.98")
            epsilon_min = params.get("epsilon_min", "0.15")
            batch_size = params.get("batch_size", "64")
            replay_buffer_size = params.get("replay_buffer_size", "100000")
            gamma = params.get("gamma", "0.99")
            learning_rate = params.get("learning_rate", "5e-4")
            learn_period = params.get("learn_period", "1")

            self.__episodes_input.setText(episodes)
            self.__seed_input.setText(seed)
            self.__epsilon_input.setText(epsilon)
            self.__epsilon_decay_rate_input.setText(epsilon_decay_rate)
            self.__epsilon_min_input.setText(epsilon_min)
            self.__batch_size_input.setText(batch_size)
            self.__replay_buffer_size_input.setText(replay_buffer_size)
            self.__gamma_input.setText(gamma)
            self.__learning_rate_input.setText(learning_rate)
            self.__learn_period_input.setText(learn_period)

            # Set the number of layers in QSpinBox according to the maximum number of layers
            layers = params.get("layers", [64])
            self.__num_layers_input.setValue(len(layers))

            if len(layers) > 0:
                neurons_layer1 = str(layers[0])
                self.__neurons_layer1_input.setText(neurons_layer1)
            if len(layers) > 1:
                neurons_layer2 = str(layers[1])
                self.__neurons_layer2_input.setText(neurons_layer2)
            if len(layers) > 2:
                neurons_layer3 = str(layers[2])
                self.__neurons_layer3_input.setText(neurons_layer3)

        # Update visibility of layer inputs
        self.__update_neurons_inputs(self.__num_layers_input.value())

    def __update_neurons_inputs(self, value):
        """
            Toggle visibility of neuron count labels and input fields based on the number of layers

            Parameters
            ----------
            value: int

            Returns
            -------
            None.

        """
        self.__layer2_label.setHidden(value < 2)
        self.__neurons_layer2_input.setHidden(value < 2)
        self.__layer3_label.setHidden(value < 3)
        self.__neurons_layer3_input.setHidden(value < 3)

    def __start_game(self):
        """
            Checks all data inputs for completeness and the correct data types.
            Subsequently, the game start is initiated.

            Parameters
            ----------
            None.

            Returns
            -------
            None.

        """

        def convert_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        def convert_int(value):
            try:
                int(value)
                return True
            except ValueError:
                return False

        error = []

        # Retrieve input values
        episodes = self.__episodes_input.text()
        seed = self.__seed_input.text()
        epsilon = self.__epsilon_input.text()
        epsilon_decay_rate = self.__epsilon_decay_rate_input.text()
        epsilon_min = self.__epsilon_min_input.text()
        batch_size = self.__batch_size_input.text()
        replay_buffer_size = self.__replay_buffer_size_input.text()
        gamma = self.__gamma_input.text()
        learning_rate = self.__learning_rate_input.text()
        learn_period = self.__learn_period_input.text()
        num_layers = self.__num_layers_input.value()
        neurons_layer1 = self.__neurons_layer1_input.text()
        neurons_layer2 = self.__neurons_layer2_input.text() if num_layers > 1 else ""
        neurons_layer3 = self.__neurons_layer3_input.text() if num_layers > 2 else ""

        # Validate inputs
        if num_layers > 1:
            if not convert_int(neurons_layer2):
                error.append(1)
        if num_layers > 2:
            if not convert_int(neurons_layer3):
                error.append(1)
        if convert_float(epsilon) and convert_float(epsilon_decay_rate) and convert_float(epsilon_min) and convert_float(gamma) and convert_float(learning_rate) and not error:
            if convert_int(episodes) and convert_int(seed) and convert_int(batch_size) and convert_int(replay_buffer_size) and convert_int(learn_period) and convert_int(neurons_layer1):
                self.__error_label.setVisible(False)
            else:
                error.append(1)
        else:
            error.append(1)

        if error:
            # Display error messages
            self.__error_label.setText("Check the data types of the inputs, not all are correct.")
            self.__error_label.setVisible(True)
        else:
            # Check correct Checkboxes
            if not (0.0 <= float(epsilon) <= 1.0) or not (0.0 <= float(epsilon_decay_rate) <= 1.0) or not (0.0 <= float(epsilon_min) <= 1.0) or not (0.0 <= float(learning_rate) <= 1.0) or not (0.0 <= float(gamma) <= 1.0):
                    self.__error_label.setText("Some input parameters are outside the permitted definition range")
                    self.__error_label.setVisible(True)
            elif not (self.__virt_cb.isChecked() or self.__proto_cb.isChecked()) or not (self.__train_cb.isChecked() or self.__eval_cb.isChecked()) or (self.__train_cb.isChecked() and not (self.__continue_train_cb.isChecked() or self.__no_continue_train_cb.isChecked())):
                self.__error_label.setText("Select at least one checkbox in each quenstion group")
                self.__error_label.setVisible(True)
            else:
                parameter = {
                    "layout": self.__plate_cb.currentText(),
                    "environment": "Virtual" if self.__virt_cb.isChecked() else "Prototype",
                    "continue_training": "Yes" if self.__continue_train_cb.isChecked() else "No",
                    "episodes": int(episodes),
                    "seed": int(seed),
                    "epsilon": float(epsilon),
                    "epsilon_decay_rate": float(epsilon_decay_rate),
                    "epsilon_min": float(epsilon_min),
                    "batch_size": int(batch_size),
                    "replay_buffer_size": int(replay_buffer_size),
                    "gamma": float(gamma),
                    "learning_rate": float(learning_rate),
                    "learn_period": int(learn_period),
                    "neurons_layer1": int(neurons_layer1),
                    "neurons_layer2": int(neurons_layer2) if neurons_layer2 else None,
                    "neurons_layer3": int(neurons_layer3) if neurons_layer3 else None,
                }
                print(parameter)

                # Close the GUI window after starting the game
                self.close()
                # Call the application_function with the collected parameters
                applicationfunction = AppFunction(parameter)
                if self.__train_cb.isChecked():
                    applicationfunction.train_main()
                else:
                    applicationfunction.evaluate_main()


    # ----------------------------------------------------------------------
    # Main
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationGUI()
    window.show()
    sys.exit(app.exec())
