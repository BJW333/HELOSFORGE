# HELOSFORGE

**Real-time 3D Model Manipulation System with AI-Powered Generation and Gesture Control**

HELOSFORGE is an advanced 3D modeling environment that combines gesture-based interaction with AI-powered model generation. Using the **TACTUS** gesture recognition system, you can manipulate 3D models with natural hand movements while leveraging OpenAI's capabilities to generate and edit STL files through natural language.

---

## Features

### Gesture Control (TACTUS Integration)
- **PINCH+MOVE**: Rotate model in 3D space
- **TWO_PINCH**: Scale model up/down
- **GRAB**: Pan/translate model
- **POINT**: Reset view to default
- **SPREAD**: Toggle wireframe mode
- **SWIPE_LEFT**: Undo (with confirmation)
- **SWIPE_RIGHT**: Redo (with confirmation)

### AI-Powered Modeling
- **Generate from scratch**: Describe a model in natural language and generate STL files
- **Edit existing models**: Modify loaded STL files with natural language instructions
- **Auto-repair**: Automatic validation and repair attempts for generated OpenSCAD code
- **OpenSCAD pipeline**: OpenAI → OpenSCAD → STL with full error handling

### Interactive 3D Viewer
- Real-time OpenGL rendering
- Wireframe/solid display modes
- Grid reference plane
- Live transformation feedback
- Undo/redo system (50 step history)
- Drag-and-drop STL loading

### File Management
- Load existing STL files
- Save transformed models
- Automatic organization (exports, generations, edits)
- Timestamped output files

---

## Requirements

### System Dependencies
- **Python 3.8+**
- **OpenSCAD**: Required for AI model generation ([Download here](https://openscad.org))
- **Webcam**: Required for gesture control

### Python Packages
```bash
pip install opencv-python numpy trimesh pygame PyOpenGL openai mediapipe
```

### Required Files
- `TACTUS_Gesture_Recognizer.py`: The TACTUS gesture recognition module
- Valid OpenAI API key

---

## Installation

1. **Clone the HELOSFORGE repository**
   ```bash
   git clone --recurse-submodules https://github.com/BJW333/HELOSFORGE.git
   cd HELOSFORGE
   ```

2. **(DO THIS ONLY IF TACTUS DOESNT POPULATE) Clone the TACTUS gesture recognition system**
   ```bash
   git clone https://github.com/BJW333/TACTUS.git
   ```
   
   After cloning, ensure `TACTUS_Gesture_Recognizer.py` is accessible to HELOSFORGE. You can either:
   - Copy `TACTUS_Gesture_Recognizer.py` from the TACTUS repo into the HELOSFORGE directory, or
   - Add the TACTUS directory to your Python path

3. **Install Python dependencies**
   ```bash
   pip install opencv-python numpy trimesh pygame PyOpenGL openai mediapipe
   ```

4. **Install OpenSCAD**
   - Download from [openscad.org](https://openscad.org)
   - Ensure it's in your system PATH or installed in the default location

5. **Configure OpenAI API Key**
   - Open `HELOSFORGE.py`
   - Replace the placeholder API key:
     ```python
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - **SECURITY WARNING**: Never commit your API key to version control. Consider using environment variables instead.

6. **Verify webcam connection**
   - Ensure your webcam is connected and accessible

---

## Usage

### Basic Launch
```bash
python HELOSFORGE.py
```

### Launch with Existing Model
```bash
python HELOSFORGE.py path/to/model.stl
```

---

## Controls Reference

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **1-5** | Load primitive shapes (cube, sphere, cylinder, cone, torus) |
| **S** | Save current model to STL |
| **G** | Generate new model from AI prompt |
| **E** | Edit current model with AI |
| **C** | Toggle camera view window |
| **W** | Toggle wireframe mode |
| **R** | Reset view to default |
| **Y** | Confirm pending action (undo/redo) |
| **N** | Cancel pending action |
| **ESC** | Quit application |

### Gesture Controls (via TACTUS)

| Gesture | Action |
|---------|--------|
| **PINCH+MOVE** | Rotate model with hand movement |
| **TWO_PINCH** | Scale model (pinch in/out) |
| **GRAB** | Pan/translate model position |
| **POINT** | Reset view to default orientation |
| **SPREAD** | Toggle between solid and wireframe |
| **SWIPE_LEFT** | Request undo (requires Y/N confirmation) |
| **SWIPE_RIGHT** | Request redo (requires Y/N confirmation) |

---

## AI Model Generation Workflow

### 1. Generate New Model (Press G)

```text
Describe the stl model you want to generate: coffee mug with handle
```

The system will:
- Send your prompt to OpenAI
- Generate OpenSCAD code
- Validate and sanitize the code
- Compile to STL via OpenSCAD
- Auto-retry up to 3 times with error feedback
- Load the result into the viewer

**Output location**: `outputs/generations/gen_TIMESTAMP.stl`

### 2. Edit Existing Model (Press E)

```text
Edit current model (e.g., 'add a 6mm hole through the center'): add a 10mm hole through the top
```

The system will:
- Save current model state as base
- Generate edit instructions in OpenSCAD
- Apply modifications using CSG operations
- Compile and load the edited model

**Output location**: `outputs/edits/edit_TIMESTAMP.stl`

---

## TACTUS Gesture Recognition System

HELOSFORGE uses **TACTUS** for robust hand gesture recognition. TACTUS provides:

- **MediaPipe-based hand tracking**: Real-time landmark detection
- **Multi-gesture classification**: Simultaneous recognition of multiple gesture types
- **Temporal filtering**: Smooth gesture transitions and debouncing
- **State management**: Gesture start/end detection with continuous tracking
- **Visual feedback**: Overlay of hand landmarks and gesture status

### TACTUS Requirements

- Webcam with clear view of hand(s)
- Adequate lighting
- Hand positioned within camera frame
- Distance: 30-60cm from camera for optimal tracking
- Try not to have more than just your hands within the frame it may cause the system to fail

---

## File Structure

```text
HELOSFORGE/
├── HELOSFORGE.py                    # Main application
├── TACTUS_Gesture_Recognizer.py     # Gesture recognition module
└── outputs/
    ├── exports/                     # Manual saves (S key)
    │   └── export_TIMESTAMP.stl
    ├── generations/                 # AI-generated models (G key)
    │   ├── gen_TIMESTAMP.stl
    │   └── gen_TIMESTAMP.scad
    └── edits/                       # AI-edited models (E key)
        ├── base_TIMESTAMP.stl
        ├── edit_TIMESTAMP.stl
        └── edit_TIMESTAMP.scad
```

---

## AI Generation Details

### Supported OpenSCAD Features

- **Primitives**: cube, sphere, cylinder, polyhedron, polygon, circle, square, text
- **Transforms**: translate, rotate, scale, mirror, linear_extrude, rotate_extrude
- **Operations**: union, difference, intersection, hull, minkowski
- **Math**: sin, cos, tan, sqrt, pow, abs, min, max, floor, ceil

### Safety Restrictions

- No external file imports (except controlled base model in edit mode)
- No external libraries (MCAD, BOSL, etc.)
- Validation and sanitization of all AI-generated code
- Automatic error recovery with contextual repair hints

### Model Specifications

- **Size**: 30-80mm (automatically scaled in viewport)
- **Quality**: $fn=64 (64 segments for curved surfaces)
- **Printability**: Generated models are designed to be 3D-printable
- **Watertight**: Enforced solid geometry validation

---

## Troubleshooting

### "OpenSCAD not found"

- Install OpenSCAD from [openscad.org](https://openscad.org)
- Ensure it's in your system PATH or installed in the default location
- macOS users: Application should be at `/Applications/OpenSCAD.app`

### "Warning: No camera"

- Check webcam connection
- Grant camera permissions to Python
- Try a different camera index (modify line 630 if needed)

### "ERROR: No OpenAI client"

- Verify API key is set correctly in line 50
- Ensure `openai` package is installed: `pip install openai`
- Check your OpenAI account has available credits

### Gestures not responding

- Toggle camera view (C key) to verify hand tracking
- Ensure adequate lighting
- Position hand 30-60cm from camera
- Make gestures clear and deliberate
- Wait for gesture state to reset between actions

### AI generation fails

- Check terminal output for specific errors
- Review generated `.scad` file in outputs directory
- Try simplifying your prompt
- Ensure OpenSCAD is properly installed

---

## Tips for Best Results

### Gesture Control

- Use deliberate, smooth movements
- Wait for gesture confirmation before next action
- Keep hand clearly visible to camera
- Use the camera view (C key) to verify tracking

### AI Prompts

- Be specific about dimensions when needed
- Describe functional features clearly
- For edits, reference existing geometry (e.g., "through the center")
- Start simple, iterate with edits

### Example Prompts

**Generation:**
- "chess pawn 40mm tall"
- "phone stand with 60 degree angle"
- "gear with 20 teeth"

**Editing:**
- "add a 5mm hole through the center"
- "round all the edges"
- "add mounting feet at the bottom"

---

## Technical Architecture

```text
┌─────────────────┐
│   User Input    │
│  (Gestures/Keys)│
└────────┬────────┘
         │
    ┌────▼─────────────┐
    │     TACTUS       │ ← Webcam feed
    │Gesture Recognizer│
    └────┬─────────────┘
         │
    ┌────▼──────────┐
    │  ModelViewer  │ ← OpenGL Rendering
    │  (Transform)  │
    └────┬──────────┘
         │
    ┌────▼──────────────┐
    │ AIModelGenerator  │ → OpenAI API
    └────┬──────────────┘
         │
    ┌────▼──────────┐
    │   OpenSCAD    │ → STL Output
    └───────────────┘
```

---

## License & Credits

**HELOSFORGE** integrates the following technologies:

- **TACTUS**: Custom gesture recognition system using MediaPipe
- **OpenAI GPT**: AI-powered model generation
- **OpenSCAD**: Solid 3D CAD modeling
- **Trimesh**: Mesh processing library
- **PyGame + PyOpenGL**: 3D visualization

Used AI for comments and to write the README everything else is my own work.

MIT License 

---

## Support

For issues, questions, or contributions:

- Check the troubleshooting section above
- Review generated `.scad` files for AI generation issues
- Ensure all dependencies are correctly installed
- Verify TACTUS gesture recognition is working via camera view

