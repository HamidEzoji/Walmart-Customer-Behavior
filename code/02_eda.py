# Plot histograms
df[['Age', 'Purchase_Amount', 'Rating']].hist(bins=20, figsize=(12, 5))
plt.tight_layout()
plt.show()

# Boxplots for outliers
for col in ['Age', 'Purchase_Amount', 'Rating']:
    plt.figure(figsize=(6, 2))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

categorical_cols = ['Gender', 'Category', 'Product_Name', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer']

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Seasonality and Temporal Patterns
df['month'] = df['Purchase_Date'].dt.month
df['day_of_week'] = df['Purchase_Date'].dt.day_name()

# Purchases by month
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='month')
plt.title('Monthly Purchase Trends')
plt.show()

# Purchases by day of week
plt.figure(figsize=(10,6))
sns.countplot(
    data=df,
    x='day_of_week',
    order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
plt.title('Day-of-Week Purchase Trends')
plt.show()

#Purpose: This boxplot visualizes the distribution of Age across Repeat Customer status,
#comparing the age of repeat customers (e.g., "Yes") versus non-repeat customers (e.g., "No").
sns.boxplot(data=df, x='Repeat_Customer', y='Age')
plt.title('Age by Repeat Customer Status')
plt.show()

#Purpose: This boxplot visualizes the distribution of Purchase Amount across Repeat Customer status,
#comparing how much repeat customers spend versus non-repeat customers.
sns.boxplot(data=df, x='Repeat_Customer', y='Purchase_Amount')
plt.title('Purchase Amount by Repeat Status')
plt.show()

# Category vs Repeat Customer
#Purpose: This plot shows the Repeat Customer rate by Product Category.
#It compares the number of repeat and non-repeat customers across various product categories.
sns.countplot(data=df, x='Category', hue='Repeat_Customer')
plt.title('Repeat Customer Rate by Product Category')
plt.xticks(rotation=45)
plt.show()

# Gender vs Repeat
#Purpose: This plot shows the Repeat Customer rate by Gender,
#comparing the number of repeat and non-repeat customers for different genders.
sns.countplot(data=df, x='Gender', hue='Repeat_Customer')
plt.title('Repeat Customer by Gender')
plt.show()

# First make sure date column is datetime type
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

# Create new time features
df['day_of_week'] = df['Purchase_Date'].dt.dayofweek  # Monday=0, Sunday=6
df['month'] = df['Purchase_Date'].dt.month
df['day'] = df['Purchase_Date'].dt.day
df['year'] = df['Purchase_Date'].dt.year

# Daily Purchases Trend
daily_purchases = df.groupby('Purchase_Date').size()

plt.figure(figsize=(15,6))
daily_purchases.plot()
plt.title('Daily Purchases Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Purchases')
plt.show()

# Weekly Pattern (Purchases by Day of the Week)
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='day_of_week', order=[0,1,2,3,4,5,6], palette='Blues')
plt.title('Purchases by Day of the Week')
plt.xlabel('Day of the Week (0=Monday)')
plt.ylabel('Number of Purchases')
plt.xticks(ticks=[0,1,2,3,4,5,6], labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

# Monthly/Seasonal Trend (Purchases by Month)
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='month', palette='Purples')
plt.title('Purchases by Month')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.show()

# Heatmap (Weekday vs Month) - Advanced!
pivot_table = df.pivot_table(index=df['month'], columns=df['day_of_week'], values='Purchase_Amount', aggfunc='count')

plt.figure(figsize=(12,8))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.0f')
plt.title('Heatmap of Purchases by Month and Day of Week')
plt.xlabel('Day of the Week (0=Monday)')
plt.ylabel('Month')
plt.show()

# Plotting purchase count and total sales side-by-side

# Group by 'Category' and aggregate purchase count and total sales amount
combined = df.groupby('Category').agg(
    purchase_count=('Purchase_Date', 'count'),  # count purchases
    Purchase_Amount=('Purchase_Amount', 'sum')   # sum purchase amounts
).reset_index()

# Select top 10 categories
top_10_combined = combined.head(10)

# Set position for each category
x = np.arange(len(top_10_combined['Category']))  # the label locations
width = 0.4  # width of the bars

# Create subplots
fig, ax1 = plt.subplots(figsize=(14,7))

# Bar for purchase count
rects1 = ax1.bar(x - width/2, top_10_combined['purchase_count'], width, label='Number of Purchases', color='skyblue')

# Create another y-axis for total sales
ax2 = ax1.twinx()

# Bar for total sales
rects2 = ax2.bar(x + width/2, top_10_combined['Purchase_Amount'], width, label='Total Sales Amount', color='lightgreen')

# Titles and labels
ax1.set_xlabel('Product Category')
ax1.set_ylabel('Number of Purchases', color='skyblue')
ax2.set_ylabel('Total Sales Amount ($)', color='lightgreen')
plt.title('Top 10 Product Categories: Number of Purchases vs Total Sales')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(top_10_combined['Category'], rotation=45, ha='right')

# Legends
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# Add data labels
for rect in rects1:
    height = rect.get_height()
    ax1.annotate('{}'.format(int(height)),
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='blue')

for rect in rects2:
    height = rect.get_height()
    ax2.annotate('${:,.0f}'.format(height),
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='green')

plt.tight_layout()
plt.show()