((style) => {
    window.videos = window.videos || {};
    const videos = [vidids];

    function onStateChange(event) {
        console.log(event.target, "is now", event.data);
        if(event.data != YT.PlayerState.PLAYING && event.data != YT.PlayerState.PAUSED) return;

        let targetState = event.data;
        let targetVideo = event.target;
        Object.keys(window.videos).forEach((videoName) => {
            let video = window.videos[videoName];
            if(video == targetVideo) return;

            let state = video.getPlayerState();
            if((state == YT.PlayerState.ENDED || state == YT.PlayerState.PAUSED) && targetState == YT.PlayerState.PLAYING) {
                video.seekTo(targetVideo.getCurrentTime());
                video.playVideo();
            }
            else if(state == YT.PlayerState.PLAYING) {
                if(targetState == YT.PlayerState.PLAYING) {
                    if(Math.abs(targetVideo.getCurrentTime() - video.getCurrentTime()) > 1.5) {
                        // Resync them
                        video.seekTo(targetVideo.getCurrentTime());
                    }
                }
                else if(targetState == YT.PlayerState.PAUSED) {
                    video.pauseVideo();
                }
            }
        });
    }

    window.onYouTubeIframeAPIReady = function() {
        Object.keys(videos).forEach((video) => {
            videoData = videos[video];
            let player = new YT.Player(video, {
                width:  videoData.width,
                height: videoData.height,
                videoId: videoData.id,
                playerVars: {
                    playsinline: 1,
                    autoplay: 1
                },
                events: {
                    onReady: (event) => {
                        console.log("Player ready");
                        setTimeout(() => event.target.playVideo(), 1000);
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

    // TODO: Hack the dcc.graphs so that they have an extra trace that shows the current video position
    /* setInterval(() => {
        for(let key in window.videos) {
            let video = window.videos[key];
            if(video.getPlayerState() == YT.PlayerState.PLAYING) {
                '<rect class="blahblah" data-subplot="xy" x="80" y="100" width="2" height="300" style="fill: black; stroke-width: 0;"></rect>';

            }
        }
    }) */

    console.log("Script inserted");
    return style;
})