var activating = false;
var stream_mode = "video";
var pause = false;
var next = 1;
var mobile = false;
var canvas
var ctx
var video;
var webcamWidth;
var webcamHeight;
canvas = document.createElement('canvas')
video = document.createElement('video')
video.setAttribute('autoplay', true)
ctx = canvas.getContext('2d')
navigator.getUserMedia = (
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia
);

$('#pause').on('click', function() {
    pause = !pause
    if (pause == true) {
        $("#status").text("Status: paused");
    } else {
        $("#status").text("Status: running");
    }
    $.post("/pause", {
        status: pause
    });
});

function getCurrentFrame() {
    if (!pause) {
        img = ctx.drawImage(video, 0, 0)
        img_dataURI = canvas.toDataURL('image/png')
        $.ajax({
            type: "POST",
            url: "/get_frame",
            data: {
                imageBase64: img_dataURI
            }
        })
    }
}

function frameLoop() {
    setTimeout(function() {
        getCurrentFrame();
        if (stream_mode == "webcam") {
            frameLoop();
        }
    }, 100)
}

async function changeBackgroundColor(btn) {
    origin_color = getComputedStyle(document.querySelector("#" + btn)).backgroundColor;
    $("#" + btn).css({
        "background": "white"
    });
    await new Promise(resolve => setTimeout(resolve, 250));
    if (btn.startsWith("activate")) {
        $("#" + btn).text("Activated")
        const last = parseInt(btn.charAt(btn.length - 1));
        for (let i = 1; i < 6; i++) {
            if (last != i) {
                $("#activate" + i).text("Activate")
            }
        }
    }
    $("#" + btn).css({
        "background": origin_color
    });
}

$("#webcam").click(function(_callback) {
    if (!mobile && $("#webcam").css("pointer-events") == "auto" && $('.switch-button-checkbox').is(':checked')) {
        $("#webcam").css("pointer-events", "none");
        if (stream_mode == "video") {
            activateLoadingStream();
            $("#status").text("Status: checking webcam");
            if (navigator.getUserMedia) {
                navigator.getUserMedia({
                        video: true,
                        audio: false
                    },

                    function(stream) {
                        webcamWidth = stream.getVideoTracks()[0].getSettings().width
                        webcamHeight = stream.getVideoTracks()[0].getSettings().height
                        canvas.setAttribute('width', webcamWidth);
                        canvas.setAttribute('height', webcamHeight);
                        video.srcObject = stream
                        stream_mode = "webcam"
                        $("#next").text("Deactivated");
                        $("#next").css("pointer-events", "none");
                        frameLoop();
                        $.post("/stream_mode", {
                            status: stream_mode
                        });
                        deactivateLoadingStream();
                    },
                    function(err) {
                        $('.switch-button-checkbox').prop('checked', false);
                        showalert("Permission is denied. You must allow access.");
                        deactivateLoadingStream();
                    }
                );
            } else {
                $('.switch-button-checkbox').prop('checked', false);
                showalert("We couldnt found a webcam.");
                deactivateLoadingStream();
            }
        }
    } else {
        loadingStream("change_mode", 1800);
        stream_mode = "video"
        $("#next").text("Next Video");
        $("#next").css("pointer-events", "auto");
        $.post("/stream_mode", {
            status: stream_mode
        });
    }
    pause = false;
    $("#webcam").css("pointer-events", "auto");
    $.post("/pause", {
        status: pause
    });
    $("#status").text("Status: running");
});

$("#next").click(function() {
    if (stream_mode == "video") {
        if (next == 4) {
            next = 1
        } else {
            next++;
        }
        $.post("/next", {
            status: next
        });
        pause = false;
        $.post("/pause", {
            status: pause
        });
    }
});

$("#info").click(function() {
    $("#cover").show();
});

$('#close-infobox').click(function() {
    $("#cover").hide();
});

function activateLoadingStream() {
    $("#stream-loading").css({
        background: "-webkit-gradient(linear, left top, left bottom, from(#FFFFFF), to(#000000))"
    });
    $("#chart-loading").css("display", "block")
    $("#chart").css("filter", "blur(10px)");
    $("#chart").css("-webkit-filter", "blur(10px)");
    $("#stream-loading").css("display", "block");
    activating = true;
}

function deactivateLoadingStream() {
    setTimeout(function() {
        $("#stream-loading").fadeOut();
        $("#chart-loading").fadeOut()
        $("#chart").css("filter", "blur(0px)");
        $("#chart").css("-webkit-filter", "blur(0px)");
        activating = false;
    }, 1000);
}

function loadingStream(btn, time) {
    if (btn == "change_mode") {
        $("#stream-loading").css({
            background: "-webkit-gradient(linear, left top, left bottom, from(#FFFFFF), to(#000000))"
        });
    } else {
        $("#stream-loading").css("background", window.getComputedStyle(document.getElementById(btn)).backgroundColor);
    }
    $("#chart-loading").css("display", "block")
    $("#chart").css("filter", "blur(10px)");
    $("#chart").css("-webkit-filter", "blur(10px)");
    $("#stream-loading").css("display", "block");
    activating = true;
    setTimeout(function() {
        $("#stream-loading").fadeOut();
        $("#chart-loading").fadeOut()
        $("#chart").css("filter", "blur(0px)");
        $("#chart").css("-webkit-filter", "blur(0px)");
        activating = false;
    }, time);
}

function showalert(message) {
    $("#popmessage").text(message);
    $('.popup-wrap').fadeIn(500);
    $('.popup-box').removeClass('transform-out').addClass('transform-in');
}
$('.btn').click(function(e) {
    $('.popup-wrap').css("display", "none");
    $('.popup-box').removeClass('transform-in').addClass('transform-out');
    e.preventDefault();
});

document.addEventListener('click', printMousePos, true);

function printMousePos(e) {
    var cursorX = e.pageX;
    var cursorY = e.pageY;
    var rect = document.getElementById("activate1").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        if (activating == true) {
            showalert("An AI Model is loading. Please be patient.");
        } else if ($("#activate1").text() == "Activated") {
            showalert("The selected AI model is already activated.");
        } else {
            $("#architecture").attr("src", "static/images/DACL.jpeg");
            $.post("/set_current_model", {
                status: "1"
            });
            $("#currentmodel").text("Current Model: DACL");
            loadingStream("activate1", 3000);
            changeBackgroundColor("activate1");
        }
    }
    var rect = document.getElementById("github1").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://github.com/SaidTogru/Facial-expression-recognition-web-application/blob/main/training/DACL.py", "_blank");
    }
    var rect = document.getElementById("paper1").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://openaccess.thecvf.com/content/WACV2021/papers/Farzaneh_Facial_Expression_Recognition_in_the_Wild_via_Deep_Attentive_Center_WACV_2021_paper.pdf", "_blank");
    }

    var rect = document.getElementById("activate2").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        if (activating == true) {
            showalert("An AI Model is loading. Please be patient.");
        } else if ($("#activate2").text() == "Activated") {
            showalert("The selected AI model is already activated.");
        } else {
            $("#architecture").attr("src", "static/images/DAN.jpeg");
            $.post("/set_current_model", {
                status: "2"
            });
            $("#currentmodel").text("Current Model: DAN");
            loadingStream("activate2", 3000);
            changeBackgroundColor("activate2");
        }
    }
    var rect = document.getElementById("github2").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://github.com/SaidTogru/Facial-expression-recognition-web-application/blob/main/training/DAN.py", "_blank");
    }
    var rect = document.getElementById("paper2").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://arxiv.org/pdf/2109.07270v4.pdf", "_blank");
    }
    var rect = document.getElementById("activate3").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        if (activating == true) {
            showalert("An AI Model is loading. Please be patient.");
        } else if ($("#activate3").text() == "Activated") {
            showalert("The selected AI model is already activated.");
        } else {
            $("#architecture").attr("src", "static/images/DeepEmotion.jpeg");
            $.post("/set_current_model", {
                status: "3"
            });
            $("#currentmodel").text("Current Model: DeepEmotion");
            loadingStream("activate3", 3000);
            changeBackgroundColor("activate3");
        }
    }
    var rect = document.getElementById("github3").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://github.com/SaidTogru/Facial-expression-recognition-web-application/blob/main/training/DeepEmotion.py", "_blank");
    }
    var rect = document.getElementById("paper3").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://arxiv.org/pdf/2109.07270v4.pdf", "_blank");

    }
    var rect = document.getElementById("activate4").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        if (activating == true) {
            showalert("An AI Model is loading. Please be patient.");
        } else if ($("#activate4").text() == "Activated") {
            showalert("The selected AI model is already activated.");
        } else {
            $("#architecture").attr("src", "static/images/BasicNet.jpeg");
            $.post("/set_current_model", {
                status: "4"
            });
            $("#currentmodel").text("Current Model: BasicNet");
            loadingStream("activate4", 3000);
            changeBackgroundColor("activate4");
        }
    }
    var rect = document.getElementById("github4").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://github.com/SaidTogru/Facial-expression-recognition-web-application/blob/main/training/BasicNet.py", "_blank");
    }
    var rect = document.getElementById("paper4").getBoundingClientRect();
    if (cursorX > rect.left && cursorX < rect.right && cursorY > rect.top && cursorY < rect.bottom) {
        window.open("https://openaccess.thecvf.com/content/WACV2021/papers/Farzaneh_Facial_Expression_Recognition_in_the_Wild_via_Deep_Attentive_Center_WACV_2021_paper.pdf", "_blank");
    }
}
/*LOADING*/
function onReady(callback) {
    var intervalId = window.setInterval(function() {
        if (document.getElementsByTagName('body')[0] !== undefined) {
            window.clearInterval(intervalId);
            callback.call(this);
        }
    }, 5000);
}

window.onresize = function(event) {
    if ($(window).width() <= 961) {
        $("#main").css("visibility", "hidden");
        $("#loading").css("display", "none");
        mobile = true;
    }
    if ($(window).width() > 961) {
        $("#main").css("visibility", "visible");
        $("#loading").css("display", "none");
        mobile = false;
    }
};

onReady(function() {
    if ($(window).width() > 961) {
        $("#main").css("visibility", "visible");
        $("#loading").fadeOut();
        mobile = false;
    }
});