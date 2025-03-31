# YOLO Coefficient Visualizations

This repository contains tools and visualizations for analyzing YOLO model coefficients and feature representations.

## Features

- Feature extraction from YOLO models
- Coefficient analysis and visualization
- Clustering analysis of feature representations
- Interactive visualizations

## Project Structure

```
yolo-coefficient-visualizations/
├── data/               # Data directory
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
├── utils/             # Utility functions
└── visualizations/    # Generated visualizations
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-coefficient-visualizations.git
cd yolo-coefficient-visualizations
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run feature extraction:
```bash
python src/extract_features.py
```

2. Generate visualizations:
```bash
python src/generate_visualizations.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 