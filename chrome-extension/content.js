// Récupérer les commentaires chargés sur YouTube
function getYouTubeComments() {
    const commentElements = document.querySelectorAll(
        "ytd-comment-thread-renderer #content-text"
    );

    const comments = [];

    commentElements.forEach(element => {
        const comment = element.innerText.trim();

        if (comment.length > 0) {
            comments.push(comment);
        }
    });

    
    return [...new Set(comments)].slice(0, 50);
}


chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "GET_COMMENTS") {
        sendResponse({
            comments: getYouTubeComments()
        });
    }
});