x = np.arange(1, 13)  # Months
y = np.array([45, 47, 52, 55, 61, 66, 
              72, 68, 65, 60, 55, 50])  # Sales


plt.figure(figsize=(6, 3))
plt.plot(x, y, "C1--o", label="Monthly Sales")
plt.title("2023 Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales (K$)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
# plt.show()
plt.savefig(
    "Basic Styling.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
