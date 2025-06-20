import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ydf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .features import (
    append_avg_price,
    append_city_avg_price,
    append_price_std,
    append_sqm_per_bed,
    append_city_avg_price_per_sqm,
    append_closest_school_gender_dummies,
    append_epc_rating_dummies,
    append_property_type_dummies,
    append_sqm_per_room,
    append_tenure_dummies,
    append_total_crime,
    append_total_rooms,
)
from .utils import (
    calculate_success_rate,
    get_train_test,
    prepare_dataset,
    save_var_importances,
)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(NeuralNet, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_forest(threshold: float, save_train_dataset: bool = False):
    df = prepare_dataset()
    df = append_sqm_per_bed(df)
    df = append_sqm_per_room(df)
    df = append_total_crime(df)
    df = append_total_rooms(df)
    # df = append_city_avg_price(df)
    df = append_city_avg_price_per_sqm(df)
    df = df[df["city"] != "london"]

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "postcode",
            "street",
            "address",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "num_schools",
            "lat",
            "lng",
            "crime_theft",
            "crime_drugs",
        ]
    )

    X_train, X_test, y_train, y_test = get_train_test(df)

    if save_train_dataset:
        df = X_train.copy()
        df["target"] = y_train
        df.to_csv("msc/train-forest.csv", index=False)

    train_df = X_train.copy()
    train_df["target"] = y_train

    learner = ydf.GradientBoostedTreesLearner(
        "target", task=ydf.Task.REGRESSION, max_depth=100, num_trees=1000
    )
    model = learner.train(train_df)
    save_var_importances(model.variable_importances())
    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train_linear_reg(threshold: float, save_train_dataset: bool = False):
    df = prepare_dataset()
    df = append_sqm_per_bed(df)
    df = append_sqm_per_room(df)
    df = append_total_crime(df)
    df = append_total_rooms(df)
    df = append_city_avg_price(df)
    df = append_city_avg_price_per_sqm(df)
    df = append_property_type_dummies(df)
    df = append_tenure_dummies(df)
    df = append_epc_rating_dummies(df)
    df = append_avg_price(df)
    df = append_price_std(df)
    df = df[df["city"] != "london"]

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "city",
            "postcode",
            "street",
            "address",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "crime_violence",
        ]
    )

    model = LinearRegression()

    X_train, X_test, y_train, y_test = get_train_test(df)

    model.fit(X_train, y_train)

    if save_train_dataset:
        df = X_train.copy()
        df["target"] = y_train
        df.to_csv("msc/train-lr.csv", index=False)

    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train_neural_net(threshold: float):
    def _compute_success_rate(predictions, actuals, threshold):
        """
        Compute success rate based on threshold.
        Success if |prediction - actual| <= threshold
        """
        predictions = (
            predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        )
        actuals = actuals.cpu().numpy() if torch.is_tensor(actuals) else actuals
        differences = np.abs(predictions - actuals)
        successes = np.sum(differences <= threshold)
        total = len(predictions)
        return successes / total

    df = prepare_dataset()
    df = df.drop(
        columns=[
            "sold_date",
            "price_str",
            "tenure",
            "property_type",
            "epc_rating",
            "address",
            "city",
            "postcode",
            "street",
            "closest_school_gender",
            "closest_poi_name",
            "closest_school_name",
        ]
    )
    X_train, X_test, y_train, y_test = get_train_test(df)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    input_size = X_train_scaled.shape[1]
    model = NeuralNet(input_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training parameters
    num_epochs = 750
    best_loss = float("inf")
    patience = 20
    patience_counter = 0

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Learning rate scheduling
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Predictions on scaled data
        train_pred_scaled = model(X_train_tensor).squeeze()
        test_pred_scaled = model(X_test_tensor).squeeze()

        # Inverse transform to original scale
        train_pred = scaler_y.inverse_transform(
            train_pred_scaled.numpy().reshape(-1, 1)
        ).flatten()
        test_pred = scaler_y.inverse_transform(
            test_pred_scaled.numpy().reshape(-1, 1)
        ).flatten()

        train_success_rate = _compute_success_rate(
            train_pred, y_train.values, threshold
        )
        test_success_rate = _compute_success_rate(test_pred, y_test.values, threshold)

        # Calculate RMSE for reference
        train_rmse = np.sqrt(np.mean((train_pred - y_train.values) ** 2))
        test_rmse = np.sqrt(np.mean((test_pred - y_test.values) ** 2))

        print(f"\n=== Results with threshold = {threshold:,.0f} ===")
        print(
            f"Training Success Rate: {train_success_rate:.3f} ({train_success_rate*100:.1f}%)"
        )
        print(
            f"Test Success Rate: {test_success_rate:.3f} ({test_success_rate*100:.1f}%)"
        )
        print(f"Training RMSE: {train_rmse:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")

        # Show some example predictions
        print(f"\n=== Sample Predictions ===")
        for i in range(min(5, len(test_pred))):
            actual = y_test.iloc[i]
            predicted = test_pred[i]
            diff = abs(predicted - actual)
            success = "✓" if diff <= threshold else "✗"
            print(
                f"{success} Actual: {actual:8.0f}, Predicted: {predicted:8.0f}, Diff: {diff:8.0f}"
            )

    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_success_rate": train_success_rate,
        "test_success_rate": test_success_rate,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "threshold": threshold,
    }


def train():
    ydf.verbose(0)
    for title, train_func in (
        # ("Forest", train_forest),
        ("Linear Regression", train_linear_reg),
        # ("Neural Network", train_neural_net),
    ):
        result = train_func(threshold=10_000, save_train_dataset=True)

        if isinstance(result, tuple):
            _, success_rate, _ = result
            print(f"{title} Success Rate: {success_rate}%")
        else:
            print(result)


if __name__ == "__main__":
    train()
