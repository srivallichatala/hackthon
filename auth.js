document.addEventListener("DOMContentLoaded", () => {
    const signinForm = document.getElementById("signinForm")
    const signupForm = document.getElementById("signupForm")
  
    if (signinForm) {
      signinForm.addEventListener("submit", handleSignIn)
    }
  
    if (signupForm) {
      signupForm.addEventListener("submit", handleSignUp)
    }
  })
  
  function handleSignIn(e) {
    e.preventDefault()
    const email = document.getElementById("email").value
    const password = document.getElementById("password").value
  
    // Client-side validation
    if (!email || !password) {
      alert("Please fill in all fields")
      return
    }
  
    // Send sign-in request to server
    fetch("/api/signin", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          window.location.href = "/"
        } else {
          alert(data.message)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        alert("An error occurred. Please try again.")
      })
  }
  
  function handleSignUp(e) {
    e.preventDefault()
    const username = document.getElementById("username").value
    const email = document.getElementById("email").value
    const password = document.getElementById("password").value
    const confirmPassword = document.getElementById("confirmPassword").value
  
    // Client-side validation
    if (!username || !email || !password || !confirmPassword) {
      alert("Please fill in all fields")
      return
    }
  
    if (password !== confirmPassword) {
      alert("Passwords do not match")
      return
    }
  
    // Send sign-up request to server
    fetch("/api/signup", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `username=${encodeURIComponent(username)}&email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          alert(data.message)
          window.location.href = "/signin"
        } else {
          alert(data.message)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        alert("An error occurred. Please try again.")
      })
  }
  
  