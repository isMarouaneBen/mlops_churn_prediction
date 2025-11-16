import pandas as pd
import numpy as np
from faker import Faker
import random
import argparse

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_ecommerce_dataset(num_records=1000):
    """
    Generate synthetic e-commerce customer dataset
    
    Parameters:
    -----------
    num_records : int
        Number of customer records to generate
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset with all required columns
    """
    
    # Define value options
    city_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    devices = ['Mobile Phone', 'Computer', 'Tablet']
    payment_modes = ['Debit Card', 'Credit Card', 'UPI', 'Cash on Delivery', 'E-wallet']
    genders = ['Male', 'Female']
    order_categories = ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others']
    marital_statuses = ['Single', 'Married', 'Divorced']
    
    data = {
        'CustomerID': [f'CUST{str(i).zfill(6)}' for i in range(1, num_records + 1)],
        'Churn': np.random.choice([0, 1], size=num_records, p=[0.7, 0.3]),
        'Tenure': np.round(np.random.uniform(0, 60, num_records), 1),
        'PreferredLoginDevice': np.random.choice(devices, num_records),
        'CityTier': np.random.choice(city_tiers, num_records),
        'WarehouseToHome': np.round(np.random.uniform(5, 127, num_records), 1),
        'PreferredPaymentMode': np.random.choice(payment_modes, num_records),
        'Gender': np.random.choice(genders, num_records),
        'HourSpendOnApp': np.round(np.random.uniform(0, 5, num_records), 1),
        'NumberOfDeviceRegistered': np.random.randint(1, 7, num_records),
        'PreferedOrderCat': np.random.choice(order_categories, num_records),
        'SatisfactionScore': np.random.randint(1, 6, num_records),
        'MaritalStatus': np.random.choice(marital_statuses, num_records),
        'NumberOfAddress': np.random.randint(1, 23, num_records),
        'Complain': np.random.choice([0, 1], size=num_records, p=[0.75, 0.25]),
        'OrderAmountHikeFromlastYear': np.round(np.random.uniform(11, 26, num_records), 1),
        'CouponUsed': np.random.randint(0, 17, num_records),
        'OrderCount': np.random.randint(1, 17, num_records),
        'DaySinceLastOrder': np.round(np.random.uniform(0, 46, num_records), 1),
        'CashbackAmount': np.round(np.random.uniform(100, 325, num_records), 2)
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main function to generate and save dataset"""
    
    parser = argparse.ArgumentParser(description='Generate E-Commerce Customer Dataset')
    parser.add_argument('-n', '--num_records', type=int, default=1000,
                        help='Number of records to generate (default: 1000)')
    parser.add_argument('-o', '--output', type=str, default='ecommerce_dataset.csv',
                        help='Output CSV filename (default: ecommerce_dataset.csv)')
    parser.add_argument('--preview', action='store_true',
                        help='Display preview of generated data')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_records} customer records...")
    df = generate_ecommerce_dataset(args.num_records)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"✓ Dataset saved to '{args.output}'")
    
    # Display statistics
    print(f"\nDataset Statistics:")
    print(f"  Total Records: {len(df)}")
    print(f"  Total Columns: {len(df.columns)}")
    print(f"  Churn Rate: {df['Churn'].mean():.2%}")
    print(f"  Complaint Rate: {df['Complain'].mean():.2%}")
    print(f"  Avg Satisfaction Score: {df['SatisfactionScore'].mean():.2f}")
    
    # Display preview if requested
    if args.preview:
        print(f"\nDataset Preview:")
        print(df.head(10).to_string())
        print(f"\nData Types:")
        print(df.dtypes)


if __name__ == "__main__":
    main()