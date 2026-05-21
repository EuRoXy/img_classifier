import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import datetime
import torch
from transformers import pipeline

# Configure matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

# 1 Configure Streamlit page settings
st.set_page_config(
    page_title="AI Image Classification Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3: Initialize session state variables
# Initialize session state for analyzed_images (empty list)
if None not in st.session_state:
    st.session_state.analyzed_images = []

# Initialize session state for model_loaded (False)
if None not in st.session_state:
    st.session_state.model_loaded = False

# Initialize session state for classifier (None)
if None not in st.session_state:
    st.session_state.classifier = None

# 4: Create model loading function with caching
# Add decorator 
@st.cache_resource
def load_image_classifier():
    """Load and cache the image classification model"""
    try:
        # Create pipeline for image classification
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        try:
            # Fallback to simpler model
            classifier = pipeline("image-classification", model="microsoft/resnet-50", device=-1)
            return classifier
        except:
            return None

# 5: Image preprocessing function
def preprocess_image(image):
    """Preprocess image for classification"""
    # Check if image mode is RGB, convert if not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# 6: Image classification function
def classify_image(image, classifier, top_k=5):
    """Classify a single image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Get predictions from classifier
        results = classifier(processed_image, top_k=top_k)
        return results
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

# 12: Create prediction visualization chart
def create_prediction_chart(predictions):
    """Create a horizontal bar chart for predictions using matplotlib"""
    if not predictions:
        return None
    
    # Extract labels and scores from predictions
    labels = [pred['label'] for pred in predictions]
    scores = [pred['score'] for pred in predictions]
    
    # Create matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart with colormap
    bars = ax.barh(labels, scores, color=plt.cm.viridis(np.array(scores)))
    
    # Customize the chart
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top Predictions')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# 16: Analytics dashboard function
def create_analytics_dashboard(analyzed_images):
    """Create analytics visualizations using matplotlib"""
    if not analyzed_images:
        st.info("No analyzed images yet. Upload and classify some images first!")
        return
    
    # Prepare data from analyzed_images
    all_predictions = []
    for img_data in analyzed_images:
        for pred in img_data['predictions']:
            all_predictions.append({
                'image_name': img_data['name'],
                'label': pred['label'],
                'score': pred['score'],
                'timestamp': img_data['timestamp']
            })
    
    df = pd.DataFrame(all_predictions)
    
    # 16: Create metrics using st.columns and st.metric
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Display total images metric
        st.metric("Total Images", len(analyzed_images))
    
    with col2:
        # Display total predictions metric
        st.metric("Total Predictions", len(df))
    
    with col3:
        avg_confidence = df['score'].mean()
        # Display average confidence metric
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        top_class = df['label'].mode().iloc[0] if not df.empty else "None"
        # Display most common class metric
        st.metric("Most Common Class", top_class)
    
    # 17 & 18: Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # 17: Top classes distribution
        class_counts = df['label'].value_counts().head(10)
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        # Create horizontal bar chart
        bars = ax1.barh(class_counts.index, class_counts.values, color='skyblue')
        ax1.set_xlabel('Count')
        ax1.set_title('Top 10 Predicted Classes')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        # Display chart with st.pyplot
        st.pyplot(fig1)
    
    with col2:
        # 18: Confidence distribution
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        # Create histogram
        ax2.hist(df['score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Display chart with st.pyplot
        st.pyplot(fig2)

# 2: Create main header and description
st.title("🖼️ AI Image Classification Dashboard")
st.markdown("""
Classify images using state-of-the-art AI models from Hugging Face. 
Upload your images to get instant predictions with confidence scores and detailed analytics.
""")

# 7: Create sidebar
with st.sidebar:
    # Add sidebar header
    st.header("🛠️ Configuration")
    
    # Add model selection
    model_option = st.selectbox(
        "Choose Model",
        [
            "google/vit-base-patch16-224 (Recommended)",
            "microsoft/resnet-50 (Faster)"
        ]
    )
    
    # Add subheader and sliders
    st.subheader("Classification Options")
    top_k = st.slider("Top K Predictions", 1, 10, 5)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1)
    
    # 7: Load model button
    if st.button("🚀 Load Model", type="primary"):
        with st.spinner("Loading AI model..."):
            # Load classifier and update session state
            st.session_state.classifier = load_image_classifier()
            st.session_state.model_loaded = True
        
        if st.session_state.classifier:
            # Show success message
            st.success("✅ Model loaded successfully!")
        else:
            # Show error message
            st.error("❌ Failed to load model")

# 8: Create main tabs
# Create three tabs
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📂 Image History", "📊 Results & Analytics"])

# 9: Single Image Tab
with tab1:
    # Add header
    st.header("Single Image Classification")
    
    # 9: File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # 10: Classification button and logic
            if st.button("🔍 Classify Image", type="primary", disabled=not st.session_state.model_loaded):
                if not st.session_state.classifier:
                    st.error("Please load the model first!")
                else:
                    with st.spinner("Classifying image..."):
                        # Classify image
                        predictions = classify_image(image, st.session_state.classifier, top_k)
                        
                        if predictions:
                            # Store results in session state
                            image_data = {
                                'name': uploaded_file.name,
                                'predictions': predictions,
                                'timestamp': datetime.now(),
                                'image': image
                            }
                            # Append to analyzed_images
                            st.session_state.analyzed_images.append(image_data)
                            
                            # 11: Display results
                            st.subheader("🎯 Predictions")
                            
                            # Loop through predictions and display with emoji indicators
                            for i, pred in enumerate(predictions):
                                confidence_color = "🟢" if pred['score'] > confidence_threshold else "🟡"
                                st.write(f"{i+1}. {confidence_color} **{pred['label']}** - {pred['score']:.2%}")
                            
                            # 12: Create and display chart
                            chart = create_prediction_chart(predictions)
                            if chart:
                                st.pyplot(chart)

# 13, 14, 15: Image History Tab
with tab2:
    # Add header
    st.header("Classification History")
    
    # 13: Check if images exist and display count
    if st.session_state.analyzed_images:
        # Display count
        st.write(f"Total classified images: {len(st.session_state.analyzed_images)}")
        
        # 14: Display previous classifications
        # Loop through images in reverse order
        for idx, img_data in enumerate(reversed(st.session_state.analyzed_images)):
            # 15: Create expander with filename and timestamp
            with st.expander(f"{img_data['name']} - {img_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display image thumbnail
                    st.image(img_data['image'], caption=img_data['name'], use_container_width=True)
                
                with col2:
                    st.write("**Top Predictions:**")
                    # Display top 3 predictions
                    for i, pred in enumerate(img_data['predictions'][:3]):
                        st.write(f"{i+1}. **{pred['label']}** - {pred['score']:.2%}")
    else:
        # Show info message
        st.info("No images classified yet. Go to the Single Image tab to start classifying!")

# 16, 17, 18, 19, 20, 21: Results & Analytics Tab
with tab3:
    # Add header
    st.header("Results & Analytics")
    # Call analytics dashboard function
    create_analytics_dashboard(st.session_state.analyzed_images)
    
    # 19, 20: Detailed results table and export
    if st.session_state.analyzed_images:
        # Add markdown header
        st.markdown("### 📋 Detailed Results")
        
        # Create detailed dataframe
        detailed_results = []
        for img_data in st.session_state.analyzed_images:
            top_pred = img_data['predictions'][0]
            detailed_results.append({
                'Image': img_data['name'],
                'Top Prediction': top_pred['label'],
                'Confidence': f"{top_pred['score']:.2%}",
                'Timestamp': img_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        results_df = pd.DataFrame(detailed_results)
        # Display dataframe
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # 20: Download functionality
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        # Add download button
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"image_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # 21: Clear results functionality
        if st.button("🗑️ Clear All Results", type="secondary"):
            # Clear analyzed_images and rerun
            st.session_state.analyzed_images = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit, Matplotlib, and Hugging Face Transformers.
</div>
""", unsafe_allow_html=True)