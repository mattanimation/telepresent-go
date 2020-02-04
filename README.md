# Telepresent GO

A client for [Telepresent](https://www.github.com/mattanimation/Telepresent) written in golang. Built on [PION](https://pion.ly).

**Note:** This is still in active development, and although is technically functional it is not reccomended to be used just yet.


## Requirements
* golang
* installation of the following until I can make this a package:
`go get {repo name}`
```
"github.com/pion/webrtc/pkg/media"
"github.com/pion/webrtc"
"github.com/gordonklaus/portaudio"
"github.com/hraban/opus"
"gocv.io/x/gocv"
"github.com/dialup-inc/ascii/vpx"
"github.com/dialup-inc/ascii/yuv"
"github.com/pion/mediadevices/pkg/codec"
"github.com/pion/mediadevices/pkg/codec/openh264"
"github.com/sacOO7/gowebsocket"
```

## Installation
Just make sure golang is installed

## Setup
* update the `config.json` file to accomodate any changes, namely the `signalServerURI`

## Usage
`go run main.go` or `go build main.go` -> then `main`


### TODO
[ ] - convert to a cleaner go package with modules
[ ] - change to using socket.io for signal server?
[ ] - other encoding options
[ ] - some computer vision features
[ ] - ROS2 integration option ? far far away
