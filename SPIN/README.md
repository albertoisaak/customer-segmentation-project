# 🎯 Customer Segmentation Analysis Platform

A professional-grade customer segmentation platform that combines traditional Machine Learning algorithms with Generative AI for comprehensive customer analysis.

## 🚀 Features

### 🔬 Traditional Machine Learning
- **K-Means Clustering**: Efficient centroid-based clustering
- **Agglomerative Clustering**: Hierarchical clustering for different cluster sizes
- **DBSCAN**: Density-based clustering with automatic outlier detection
- **Comprehensive Metrics**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **Automatic Parameter Optimization**: Smart parameter selection and validation

### 🧠 Generative AI (Gen AI)
- **Semantic Segmentation**: Behavior pattern analysis without traditional algorithms
- **OpenAI Integration**: GPT-3.5-turbo and GPT-4 support
- **Business Insights**: Automated business recommendations and strategies
- **Contextual Analysis**: Industry-specific insights and recommendations
- **Automated Reporting**: AI-generated comprehensive reports

### 📊 Advanced Analytics
- **Automatic Variable Selection**: Smart feature selection based on column patterns
- **Data Preprocessing**: Robust handling of missing values and data types
- **Real-time Visualization**: Interactive plots with Plotly
- **Comparative Analysis**: AI-powered comparison between ML and Gen AI results
- **Export Capabilities**: Download reports in Markdown format

## 🏗️ Architecture

The application follows a modular, professional architecture:

```
├── core/                    # Core application modules
│   ├── __init__.py         # Module initialization
│   ├── analyzer.py         # Main SegmentationAnalyzer class
│   ├── ml_analyzer.py      # Traditional ML functionality
│   ├── ai_analyzer.py      # Gen AI functionality
│   └── config.py           # Configuration and utilities
├── streamlit_app_refactored.py  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### 🔧 Core Modules

#### `SegmentationAnalyzer`
The main orchestrator class that handles:
- Data loading and preprocessing
- Variable selection and validation
- Coordination between ML and AI analyzers
- Result management and comparison

#### `MLAnalyzer`
Handles all traditional ML operations:
- Clustering algorithm implementation
- Metric calculation and validation
- Report generation for ML results
- Business question answering

#### `AIAnalyzer`
Manages Gen AI operations:
- OpenAI API integration
- Semantic analysis generation
- Business insight creation
- Comparative analysis

#### `Config`
Centralized configuration management:
- Application settings
- Algorithm parameters
- Error and success messages
- Utility functions

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for Gen AI features)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd customer-segmentation-platform
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run streamlit_app_refactored.py
```

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
openai>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🎯 Usage

### 1. Data Upload
- Upload a CSV file through the sidebar
- Data is automatically preprocessed
- Variables are automatically selected based on naming patterns

### 2. Traditional ML Analysis
- Select clustering algorithm (K-Means, Agglomerative, DBSCAN)
- Configure number of clusters and random state
- Click "Ejecutar ML Tradicional" to run analysis
- View metrics, cluster distribution, and generate reports

### 3. Gen AI Analysis
- Enter your OpenAI API key
- Select AI model (GPT-3.5-turbo or GPT-4)
- Adjust temperature for creativity level
- Click "Ejecutar Gen AI" to run semantic analysis
- View AI-generated insights and recommendations

### 4. Comparative Analysis
- Run both ML and Gen AI analyses
- Use the automatic comparison feature
- Download comprehensive comparison reports

## 🔍 Key Differences: ML vs Gen AI

| Aspect | Traditional ML | Gen AI |
|--------|----------------|--------|
| **Method** | Mathematical algorithms | Semantic pattern analysis |
| **Parameters** | Clusters, random state | Temperature only |
| **Output** | Statistical metrics | Business insights |
| **Validation** | Mathematical metrics | Contextual understanding |
| **Use Case** | Statistical validation | Business strategy |

## 📊 Data Requirements

### Supported Formats
- CSV files with headers
- Minimum 10 rows of data
- Mix of numeric and categorical variables

### Automatic Variable Selection
The system automatically selects relevant variables based on:
- **Demographic**: age, gender, income
- **Transactional**: purchase, transaction, ticket
- **Temporal**: day, month, frequency
- **Amount**: total, average, sum

### Excluded Variables
Automatically excluded:
- Identifiers (ID, user_id, customer_id)
- Personal info (name, email, phone)
- Geographic (address, city, country)
- Dates and timestamps

## 🎨 Customization

### Adding New Algorithms
1. Extend `MLAnalyzer` class
2. Implement clustering method
3. Add to configuration
4. Update UI options

### Customizing AI Prompts
1. Modify prompts in `AIAnalyzer`
2. Adjust temperature settings
3. Customize business context
4. Add industry-specific insights

### Styling and UI
1. Modify Streamlit configuration
2. Update color schemes in `Config`
3. Customize plot themes
4. Add new visualization types

## 🔒 Security

- API keys are handled securely
- No data is stored permanently
- All processing happens locally
- OpenAI API calls are encrypted

## 🐛 Troubleshooting

### Common Issues

1. **"Datos no preprocesados"**
   - Ensure CSV file is uploaded
   - Check data format and structure

2. **"No se encontraron variables numéricas"**
   - Verify data types in CSV
   - Check column naming patterns

3. **"API key inválida"**
   - Verify OpenAI API key format
   - Ensure key has sufficient credits

4. **"Error en Gen AI"**
   - Check internet connection
   - Verify API key permissions
   - Try different temperature settings

## 📈 Performance

### Optimization Tips
- Use smaller datasets for testing
- Adjust temperature for faster responses
- Limit number of clusters for ML
- Use GPT-3.5-turbo for faster AI analysis

### Scalability
- Handles datasets up to 10,000 rows efficiently
- Supports up to 50 variables
- Optimized for cloud deployment
- Memory-efficient processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow coding standards
4. Add comprehensive tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API access
- Streamlit for the web framework
- Scikit-learn for ML algorithms
- Plotly for interactive visualizations

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation
- Contact the development team

---

**Built with ❤️ for professional customer segmentation analysis**