# Use Python base image
FROM python:3.9-slim

# Install necessary dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project code to container
COPY . /app
WORKDIR /app

# Set environment variable for API key if needed
# ENV ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Run the script for updating data, training the agent, and visualizing results
CMD ["sh", "-c", "python hybrid_reinforcement_regression/realtime_data_fetcher.py && python hybrid_reinforcement_regression/train_regression_model.py && python hybrid_reinforcement_regression/predict_with_regression_model.py && python hybrid_reinforcement_regression/realtime_update_and_train_agent.py && python hybrid_reinforcement_regression/visualize_agent_results.py && python hybrid_reinforcement_regression/predict_next_day.py"]