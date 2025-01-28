document.addEventListener("DOMContentLoaded", () => {
    // Check if user is authenticated
    fetch("/api/check_auth")
      .then((response) => response.json())
      .then((data) => {
        if (!data.authenticated) {
          window.location.href = "/signin"
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        alert("An error occurred. Please try again.")
      })
  
    const patientForm = document.getElementById("patientForm")
    const resultsSection = document.getElementById("results")
    const signoutBtn = document.getElementById("signoutBtn")
  
    signoutBtn.addEventListener("click", () => {
      fetch("/api/signout")
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            window.location.href = "/signin"
          } else {
            alert("Error signing out. Please try again.")
          }
        })
        .catch((error) => {
          console.error("Error:", error)
          alert("An error occurred. Please try again.")
        })
    })
  
    patientForm.addEventListener("submit", function (e) {
      e.preventDefault()
      console.log("Form submitted")
  
      const formData = new FormData(this)
  
      // Show loading state
      const submitButton = this.querySelector("button")
      submitButton.disabled = true
      submitButton.textContent = "Analyzing..."
  
      fetch("/api/process_ecg", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          console.log("Response received:", response)
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
          }
          return response.json()
        })
        .then((responseData) => {
          console.log("Data received:", responseData)
          if (responseData.success) {
            displayResults(responseData.data)
          } else {
            throw new Error(responseData.message || "An unknown error occurred")
          }
        })
        .catch((error) => {
          console.error("Error:", error)
          alert("An error occurred while processing the ECG: " + error.message)
        })
        .finally(() => {
          submitButton.disabled = false
          submitButton.textContent = "Analyze ECG"
        })
    })
  
    function displayResults(data) {
      console.log("Displaying results:", data)
  
      // Display heart rate
      document.getElementById("heartRate").textContent = data.heartRate ? `${data.heartRate} BPM` : "N/A"
  
      // Display heart beat
      document.getElementById("heartBeat").textContent = data.heartBeat || "N/A"
  
      // Display risk level
      document.getElementById("riskLevel").textContent = data.riskLevel || "N/A"
  
      // Display doctor consultation
      document.getElementById("doctorConsult").textContent = data.doctorConsult || "N/A"
  
      // Create and display the stages chart
      if (data.stages) {
        createStagesChart(data.stages)
      }
  
      // Display recommendations
      if (data.recommendations) {
        displayRecommendations(data.recommendations)
      }
  
      // Show results section
      resultsSection.classList.remove("hidden")
      resultsSection.classList.add("fade-in")
      resultsSection.scrollIntoView({ behavior: "smooth" })
    }
  
    function createStagesChart(stages) {
      const ctx = document.createElement("canvas")
      ctx.id = "stagesChart"
      document.getElementById("pie-chart-container").innerHTML = ""
      document.getElementById("pie-chart-container").appendChild(ctx)
  
      new Chart(ctx, {
        type: "pie",
        data: {
          labels: Object.keys(stages),
          datasets: [
            {
              data: Object.values(stages),
              backgroundColor: ["rgba(0, 255, 255, 0.8)", "rgba(255, 0, 255, 0.8)"],
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: "right",
              labels: {
                color: "#fff",
              },
            },
            title: {
              display: true,
              text: "ECG Analysis Stages",
              color: "#fff",
            },
          },
        },
      })
    }
  
    function displayRecommendations(recommendations) {
      const exerciseRecs = document.getElementById("exerciseRecs")
      const dietRecs = document.getElementById("dietRecs")
      const lifestyleRecs = document.getElementById("lifestyleRecs")
  
      exerciseRecs.innerHTML = ""
      dietRecs.innerHTML = ""
      lifestyleRecs.innerHTML = ""
  
      if (recommendations.exercise) {
        recommendations.exercise.forEach((rec) => {
          const li = document.createElement("li")
          li.textContent = rec
          exerciseRecs.appendChild(li)
        })
      }
  
      if (recommendations.diet) {
        recommendations.diet.forEach((rec) => {
          const li = document.createElement("li")
          li.textContent = rec
          dietRecs.appendChild(li)
        })
      }
  
      if (recommendations.lifestyle) {
        recommendations.lifestyle.forEach((rec) => {
          const li = document.createElement("li")
          li.textContent = rec
          lifestyleRecs.appendChild(li)
        })
      }
    }
  })
  
  