"""
setup_db.py — Initialize and seed the SQLite database with realistic business data.

Run once before starting the application:
    python scripts/setup_db.py
"""
import sqlite3
import random
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path("data/sample_db/business.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

REGIONS    = ['North', 'South', 'East', 'West', 'Central']
SEGMENTS   = ['Enterprise', 'SMB', 'Consumer']
STATUSES   = ['completed', 'pending', 'cancelled', 'refunded']
ST_WEIGHTS = [0.70, 0.15, 0.10, 0.05]
DEPTS      = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']

PRODUCTS = [
    (1, 'Analytics Pro',   'Software', 4950, 87),
    (2, 'DataStream SDK',  'Software', 4900, 143),
    (3, 'CloudSync Suite', 'Services', 4720, 56),
    (4, 'ML Toolkit',      'Software', 4900, 201),
    (5, 'SecureVault',     'Security', 3200, 94),
    (6, 'DataPipeline',    'Software', 3800, 118),
    (7, 'API Gateway Pro', 'Services', 2900, 67),
    (8, 'EdgeDeploy',      'Hardware', 6500, 34),
]

CUSTOMER_NAMES = [
    'Acme Corp','ByteWave Inc','Nexus Systems','Orion Tech','Pinnacle LLC',
    'Redline Corp','Synapse AI','TerraVault','Uplift Digital','Vertex Corp',
    'Ironclad Labs','Luminary Data','Cascade Tech','Foundry AI','Meridian Systems',
    'Quantum Bridge','Axon Networks','BlueShift IO','CoreLogic','DataForge',
    'Elysian Cloud','FusionEdge','GridSense','HorizonX','InfiniteDB',
    'JetStream Analytics','Keystone AI','Lithic Systems','Momentum Data','Northstar Tech',
]

EMPLOYEE_DATA = [
    ('Priya Sharma',   'Engineering', 128000, '2020-04-01'),
    ('Marcus Chen',    'Sales',        94000, '2021-07-15'),
    ('Sofia Reyes',    'Marketing',    88000, '2022-01-10'),
    ('James Okafor',   'Engineering', 142000, '2019-09-23'),
    ('Anika Patel',    'HR',           76000, '2023-03-07'),
    ('Leo Nakamura',   'Finance',      98000, '2020-11-18'),
    ('Chloe Dubois',   'Engineering', 135000, '2021-02-28'),
    ('Ravi Iyer',      'Sales',        91000, '2022-06-14'),
    ('Amara Osei',     'Marketing',    82000, '2022-09-01'),
    ('Tyler Brooks',   'Engineering',  97000, '2023-01-15'),
    ('Yuki Tanaka',    'Finance',     106000, '2021-05-20'),
    ('Fatima Al-Amin', 'HR',           79000, '2022-11-08'),
    ('Ethan Walsh',    'Sales',       118000, '2020-08-12'),
    ('Nadia Petrov',   'Engineering', 161000, '2018-06-30'),
    ('Carlos Rivera',  'Marketing',    95000, '2021-10-25'),
]


def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def create_tables(conn):
    conn.executescript("""
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS employees;

    CREATE TABLE customers (
        customer_id   INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        email         TEXT UNIQUE,
        segment       TEXT,
        region        TEXT,
        created_at    DATE
    );

    CREATE TABLE products (
        product_id    INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        category      TEXT,
        price         REAL,
        stock_quantity INTEGER
    );

    CREATE TABLE orders (
        order_id      INTEGER PRIMARY KEY,
        customer_id   INTEGER REFERENCES customers(customer_id),
        product_id    INTEGER REFERENCES products(product_id),
        quantity      INTEGER,
        amount        REAL,
        order_date    DATE,
        region        TEXT,
        status        TEXT
    );

    CREATE TABLE employees (
        employee_id   INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        department    TEXT,
        salary        REAL,
        hire_date     DATE
    );
    """)
    print("✅ Tables created")


def seed(conn):
    start_date = date(2023, 1, 1)
    end_date   = date(2024, 9, 30)

    # Customers
    customers = []
    for i, name in enumerate(CUSTOMER_NAMES, 1):
        slug = name.lower().replace(' ', '').replace(',', '')
        customers.append((
            i, name, f"{slug}@example.com",
            random.choice(SEGMENTS), random.choice(REGIONS),
            rand_date(date(2019, 1, 1), date(2023, 6, 30)).isoformat(),
        ))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
    print(f"  ✅ {len(customers)} customers")

    # Products
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
    print(f"  ✅ {len(PRODUCTS)} products")

    # Orders — 1200 across 2023–2024
    orders = []
    for i in range(1, 1201):
        cust_id  = random.randint(1, len(customers))
        prod     = random.choice(PRODUCTS)
        qty      = random.randint(1, 20)
        amount   = round(prod[3] * qty, 2)
        orders.append((
            i, cust_id, prod[0], qty, amount,
            rand_date(start_date, end_date).isoformat(),
            random.choice(REGIONS),
            random.choices(STATUSES, weights=ST_WEIGHTS)[0],
        ))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)
    print(f"  ✅ {len(orders)} orders")

    # Employees
    employees = [(i+1, *e) for i, e in enumerate(EMPLOYEE_DATA)]
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)
    print(f"  ✅ {len(employees)} employees")

    conn.commit()


if __name__ == "__main__":
    print(f"Initialising database at: {DB_PATH.absolute()}\n")
    with sqlite3.connect(DB_PATH) as conn:
        create_tables(conn)
        seed(conn)
    print(f"\n🎉 Database ready → {DB_PATH.absolute()}")
    print("   Tables: customers, products, orders, employees")
