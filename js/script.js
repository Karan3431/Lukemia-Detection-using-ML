document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
function showFeatures() {
    // Find the features section by its ID
    const featuresSection = document.getElementById("features");

    // Make the section visible by changing its style
    featuresSection.style.display = "block";

    // Scroll smoothly to the section
    featuresSection.scrollIntoView({ behavior: "smooth" });
}
async function uploadImage() {
    const input = document.getElementById('imageUpload');
    const imageContainer = document.getElementById('imageContainer');
    const result = document.getElementById('result');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const progressBar = progressBarContainer.querySelector('#progressBar div');

    if (!input.files.length) {
        result.textContent = 'Please upload an image first.';
        return;
    }

    const formData = new FormData();
    formData.append('file', input.files[0]);

    result.innerHTML = '';
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '0%';

    progressBarContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

    try {
        await animateProgress(progressBar, 0, 30);

        const response = await fetch('http://127.0.0.1:5000/detect', {
            method: 'POST',
            body: formData
        });

        await animateProgress(progressBar, 30, 60);

        if (!response.ok) {
            throw new Error('Error processing image');
        }

        const data = await response.json();

        await animateProgress(progressBar, 60, 90);

        // Display the uploaded and result images
        const uploadedImgWrapper = document.createElement('div');
        uploadedImgWrapper.className = 'image-wrapper';
        const uploadedImg = new Image();
        uploadedImg.src = data.uploaded_image;
        uploadedImg.alt = 'Uploaded Image';
        uploadedImgWrapper.appendChild(uploadedImg);
        uploadedImgWrapper.innerHTML += '<div class="image-label">Uploaded Image</div>';

        const resultImgWrapper = document.createElement('div');
        resultImgWrapper.className = 'image-wrapper';
        const resultImg = new Image();
        resultImg.src = data.result_image;
        resultImg.alt = 'Result Image';
        resultImgWrapper.appendChild(resultImg);
        resultImgWrapper.innerHTML += '<div class="image-label">Result Image</div>';

        imageContainer.innerHTML = '';
        imageContainer.appendChild(uploadedImgWrapper);
        imageContainer.appendChild(resultImgWrapper);

        await animateProgress(progressBar, 90, 100);
        progressBarContainer.style.display = 'none';
        result.textContent = 'Analysis complete!';
    } catch (error) {
        result.textContent = `Error: ${error.message}`;
        progressBarContainer.style.display = 'none';
    }
}