import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage
from admin_dashboard import admin_portal  # Importing the dashboard
import time

# Function to validate login
def validate_login(username, password, login_window):
    if username == "admin" and password == "admin":
        login_window.destroy()
        admin_portal()  # Navigate to the admin portal
    else:
        messagebox.showerror("Login Failed", "Invalid Credentials")

# Function to create the admin login portal
def create_admin_portal():
    # Main login window
    login_window = tk.Tk()
    login_window.title("Admin Login")
    login_window.geometry("400x350")
    login_window.configure(bg="#e3e3e3")  # Body background color

    # Title Section
    title_frame = tk.Frame(login_window, bg="#4CAF50")  # Title background
    title_frame.pack(fill="x")
    title_label = tk.Label(
        title_frame,
        text="SMS Spam Detection using Machine Learning and Deep Learning Techniques",
        font=("Arial", 18, "bold"),
        bg="#4CAF50",  # Title background matches frame
        fg="white",
        padx=20,
        pady=20,
        wraplength=1100,
        justify="center"
    )
    title_label.pack()

    # Animation for the title
    def animate_title():
        text = "SMS Spam Detection using Machine Learning and Deep Learning Techniques"
        title_label.config(text="")  # Start with an empty text
        for i in range(1, len(text) + 1):
            title_label.config(text=text[:i])  # Gradually show characters
            login_window.update()
            time.sleep(0.05)

    # Run animation after the window is fully initialized
    login_window.after(100, animate_title)

    # Login Page UI
    login_label = tk.Label(
        login_window, 
        text="Admin Login", 
        font=("Arial", 20, "bold"), 
        bg="#e3e3e3", 
        fg="#333"
    )
    login_label.pack(pady=20)

    username_label = tk.Label(login_window, text="Username:", font=("Arial", 16), bg="#e3e3e3", fg="#333")
    username_label.pack(pady=5)
    username_entry = tk.Entry(login_window, font=("Arial", 16))
    username_entry.pack(pady=5)

    password_label = tk.Label(login_window, text="Password:", font=("Arial", 16), bg="#e3e3e3", fg="#333")
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_window, font=("Arial", 16), show="*")
    password_entry.pack(pady=5)

    login_button = tk.Button(
        login_window, 
        text="Login", 
        font=("Arial", 12, "bold"), 
        bg="#4CAF50", 
        fg="white", 
        command=lambda: validate_login(username_entry.get(), password_entry.get(), login_window)
    )
    login_button.pack(pady=20)

    login_window.mainloop()

# Run the portal
if __name__ == "__main__":
    create_admin_portal()
