((style) => {
    window.videos = window.videos || {};
    const videos = [vidids];

    function onStateChange(event) {

    }

    window.onYouTubeIframeAPIReady = function() {
        Object.keys(videos).forEach((video) => {
            videoData = videos[video];
            let player = new YT.Player(video, {
                width:  videoData.width,
                height: videoData.height,
                videoId: videoData.id,
                playerVars: {
                    playsinline: 1
                },
                events: {
                    onReady: (event) => {
                        console.log("Player ready");
                        event.target.playVideo();
                    },
                    onStateChange: onStateChange
                }
            });

            window.videos[video] = player;
        });
    };

    if(!window.ytLoaded) {
        window.ytLoaded = true;

        var tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        var firstScript = document.getElementsByTagName("script")[0];
        console.log(firstScript);
        firstScript.parentNode.insertBefore(tag, firstScript);
    }

    console.log("Script inserted");
    return style;
})