// +build gocv

package main

import (
	"fmt"
	//"bufio"
	"context"
	"encoding/json"
	//"io"
	//"io/ioutil"
	"os"
	"path"
	"strings"
	"sync"
	"image"
	"sync/atomic"
	"log"
	"os/signal"
	"time"
	"math/rand"

	"github.com/pion/webrtc/pkg/media"
	"github.com/pion/webrtc"
	
	//audio
	"github.com/gordonklaus/portaudio"
	"github.com/hraban/opus"
	
	//video
	//"github.com/hybridgroup/mjpeg"
	"gocv.io/x/gocv"
	//"github.com/dialup-inc/ascii/camera"
	"github.com/dialup-inc/ascii/vpx"
	"github.com/dialup-inc/ascii/yuv"
	//"github.com/gen2brain/x264-go"
	
	//"github.com/giorgisio/goav/avformat"
	//"github.com/pion/mediadevices"
	h264c "github.com/pion/mediadevices/pkg/codec"
	"github.com/pion/mediadevices/pkg/codec/openh264"
	//ggg "github.com/notedit/gstreamer-go"

	//web
	//"github.com/gorilla/websocket"
	"github.com/sacOO7/gowebsocket"
	//"encoding/json"
	
)


// signal server message structs
type SignaServerMessage struct {
	Message string `json:"message"`
    Name string  `json:"name"`
	Type string `json:"type"`
	Success bool `json:"success"`
	Offer webrtc.SessionDescription `json:"offer"`
	Candidate webrtc.ICECandidateInit `json:"candidate"`
}

type LoginStruct struct {
    Name string  `json:"name"`
	Type string `json:"type"`
}

type OfferStruct struct {
    Name string  `json:"name"`
	Type string `json:"type"`
	Sdp string `json:"sdp"`
}

type AnswerStruct struct {
	Name string  `json:"name"`
	Type string `json:"type"`
	Answer webrtc.SessionDescription `json:"answer"`
}

type DataChannelHandler func([]byte)

type GamepadState struct {
	FACE_1 int `json:"FACE_1"`
	FACE_2 int `json:"FACE_2"`
	FACE_3 int `json:"FACE_3"`
	FACE_4 int `json:"FACE_4"`

	LEFT_TOP_SHOULDER     int `json:"LEFT_TOP_SHOULDER"`
	RIGHT_TOP_SHOULDER    int `json:"RIGHT_TOP_SHOULDER"`
	LEFT_BOTTOM_SHOULDER  int `json:"LEFT_BOTTOM_SHOULDER"`
	RIGHT_BOTTOM_SHOULDER int`json:"RIGHT_BOTTOM_SHOULDER"`

	SELECT_BACK   int `json:"SELECT_BACK"`
	START_FORWARD int `json:"START_FORWARD"`
	LEFT_STICK    int `json:"LEFT_STICK"`
	RIGHT_STICK   int `json:"RIGHT_STICK"`
	DPAD_UP       int `json:"DPAD_UP"`
	DPAD_DOWN     int `json:"DPAD_DOWN"`

	DPAD_LEFT     int `json:"DPAD_LEFT"`
	DPAD_RIGHT    int `json:"DPAD_RIGHT"`
	HOME          int `json:"HOME"`
	LEFT_STICK_X  float32 `json:"LEFT_STICK_X"`
	LEFT_STICK_Y  float32 `json:"LEFT_STICK_Y"`
	RIGHT_STICK_X float32 `json:"RIGHT_STICK_X"`
	RIGHT_STICK_Y float32 `json:"RIGHT_STICK_Y"`
}

type GamepadData struct {
	ID string `json:"id"`
	State GamepadState `json:"state"`
	Timestamp int `json:"ts"`
}

type VideoCaptureSettings struct {
	Width int `json:"width"`
	Height int `json:"height"`
	Framerate uint8 `json:"framteRate"`
	DeviceID interface{} `json:"deviceID"`
	EncoderName string `json:"encoderName"`
}

type VideoFrameCallback func(image.Image, []byte, error)
type AudioFrameCallback func([]int16, error)


type VideoCaptureCamera struct {
	cam *gocv.VideoCapture
	callback VideoFrameCallback
	Settings VideoCaptureSettings
	IsActive bool
	mu sync.Mutex
}

func (c *VideoCaptureCamera)Init() (chan int) {
	log.Println("Initializing Camera")
	
	log.Println("ready to go")
	ch := make(chan int)
	// goroutine for capturing frames
	go func(){
		fmt.Printf("deviceid type %T\n", c.Settings.DeviceID)
		c.cam, err = gocv.OpenVideoCapture(c.Settings.DeviceID.(string))
		if err != nil {
			log.Printf("Error opening video capture device: %v\n", c.Settings.DeviceID)
			return 
		}
		defer c.cam.Close()
		c.cam.Set(gocv.VideoCaptureFrameWidth, float64(c.Settings.Width))
		c.cam.Set(gocv.VideoCaptureFrameHeight, float64(c.Settings.Height))

		img := gocv.NewMat()
		defer img.Close()

		window := gocv.NewWindow("CV Window")
		defer window.Close()

		for {
			//log.Println("in loop")
			if c.IsActive {
				//log.Println("is active")
				//c.mu.Lock()
				if ok := c.cam.Read(&img); !ok {
					log.Printf("Device closed: %v\n", c.Settings.DeviceID)
					break 
				}
				if img.Empty() {
					continue
				}
				//make as ycbr image
				i, err := img.ToImage()
				chk(err)
				

				window.IMShow(img)
				if window.WaitKey(1) == 27 {
					//vw.Close()
					break
				}

				if c.callback == nil{
					continue
				}
				c.callback(i, img.ToBytes(), nil)
				//c.mu.Unlock()
			}
			//handle case to stop goroutine
			select {
			case <-ch:
				return
			default:
				continue
			}
		}
	}()

	return ch
}

func (c *VideoCaptureCamera)Start(){
	log.Println("Starting Camera")
	c.mu.Lock()
	c.IsActive = true
	c.mu.Unlock()
}

func (c *VideoCaptureCamera)Stop(){
	log.Println("Shutting Down Camera")
	c.mu.Lock()
	c.IsActive = false
	c.mu.Unlock()
}

func NewVideoCaptureCamera(settings VideoCaptureSettings)(VideoCaptureCamera, error){

	return VideoCaptureCamera { Settings: settings, IsActive: false,}, nil
}


type VideoCapture struct {
	vpxEnc *vpx.Encoder
	x264Enc h264c.VideoEncoder
	EncoderName string
	Input VideoCaptureCamera
	inputChannel chan int

	pts   int
	mux sync.Mutex

	buffer []byte
	forceKeyframe uint32
	encodeLock    uint32

	track *webrtc.Track
}

func (c *VideoCapture) Init() error {
	log.Println("Init Video Capture")
	c.mux.Lock()
	c.Input.callback = c.onFrame
	c.mux.Unlock()
	c.inputChannel = c.Input.Init()
	c.SetEncoder(c.EncoderName)
	return nil
}

func (c *VideoCapture) Start() error {
	log.Println("Started Capture")
	c.Input.Start()
	return nil
}

func (c *VideoCapture) Stop() error {
	log.Println("Stopped Capture")
	close(c.inputChannel)
	c.Input.Stop()
	return nil
}

func (c *VideoCapture) RequestKeyframe() {
	atomic.StoreUint32(&c.forceKeyframe, 1)
}

func (c *VideoCapture) SetTrack(track *webrtc.Track) {
	c.mux.Lock()
	c.track = track
	c.mux.Unlock()
}

func (c *VideoCapture) SetEncoder(encoderName string) {
	log.Println("Attempting to set encoder to: ", encoderName)
	switch encoderName {
	case "VP8":
		c.vpxEnc, err = vpx.NewEncoder(c.Input.Settings.Width, c.Input.Settings.Height)
		chk(err)
	case "H264":
		selectedResolution := "480p"
		bitrateMap := map[string]float32{"720p": 2.56, "480p": 1.28, "360p": 0.96} //res name : common bitrate mbps
		targetBits := int((float32(c.Input.Settings.Width * c.Input.Settings.Height) * bitrateMap[selectedResolution]) / 1000)
		log.Println("tb", targetBits)
		encSett := h264c.VideoSetting{
			Width:     c.Input.Settings.Width,
			Height:    c.Input.Settings.Height,
			TargetBitRate: targetBits,
			MaxBitRate:    2500,
			FrameRate: float32(c.Input.Settings.Framerate),
		}
		log.Println("encoder settings: ", encSett)
		c.x264Enc, err = openh264.NewEncoder(encSett)
		chk(err)
	}
	c.EncoderName = encoderName
}

func (c *VideoCapture)encodeFrame(img image.Image, buffer []byte, pts int, forceKeyframe bool) (int, error)  {
	// TODO handle different encoders like VP8 and H264
	//log.Println("encoding frame with: ", c.EncoderName)
	switch c.EncoderName {
	case "VP8", "VP9":
		n, err := c.vpxEnc.Encode(buffer, img, c.pts, forceKeyframe)
		return n, err
	case "H264":
		// h264 below
		//gocv.CvtColor(img, &img, gocv.ColorBGRToYUV)
		//goimg, err := img.ToImage()
		//chk(err)

		//tmpimg := x264.NewYCbCr(goimg.Rect)
		//yimg := tmpimg.ToYCbCr(goimg)
		yimgb, w, h := yuv.ToI420(img)
		yimg, err := yuv.FromI420(yimgb, w, h)
		//yimg := image.NewYCbCr(img.Bounds(), image.YCbCrSubsampleRatio420)

		//encoder requires image to be YCbCr and 4:2:0 format
		buff, err := c.x264Enc.Encode(yimg)
		for i := range buff {
			buffer[i] = buff[i]
		}
		return len(buff), err
	}
	
	return 0, nil
}

func (c *VideoCapture) onFrame(img image.Image, vBuff []byte, err error) {
	if err != nil {
		log.Println("err: ", err)
		return
	}
	//log.Println("OnFrame is running")

	if !atomic.CompareAndSwapUint32(&c.encodeLock, 0, 1) {
		return
	}
	defer atomic.StoreUint32(&c.encodeLock, 0)

	forceKeyframe := atomic.CompareAndSwapUint32(&c.forceKeyframe, 1, 0)

	//log.Println("About to call encode frame: ", img, len(vBuff), c.pts, forceKeyframe)
	n, err := c.encodeFrame(img, vBuff, c.pts, forceKeyframe)
	if err != nil {
		log.Println("encode: ", err)
		return
	}
	c.pts++
	//log.Println(n)
	data := vBuff[:n]
	samp := media.Sample{Data: data, Samples: 1}

	if c.track == nil {
		return
	}

	c.mux.Lock()
	//log.Println("writing sample: ", samp)
	if err := c.track.WriteSample(samp); err != nil {
		log.Println("write sample: ", err)
		return
	}
	c.mux.Unlock()
}

func NewVideoCapture(settings VideoCaptureSettings) (*VideoCapture, error) {
	
	cam, err := NewVideoCaptureCamera(settings)
	chk(err)
	vcap := &VideoCapture { Input: cam, EncoderName: settings.EncoderName, }
	return vcap, nil
}


type AudioCaptureSettings struct {
	SampleRate int `json:"sampleRate"`
	Channels int `json:"channels"`
	Latency float32 `json:"latency"`
	EncoderName string `json:"encoderName"`
}

type AudioCapture struct {
	opusEnc *opus.Encoder
	mic *portaudio.Stream
	inputChannel chan int
	Settings AudioCaptureSettings
	sampleSize int
	buffer []int16
	forceKeyframe uint32
	encodeLock    uint32
	callback AudioFrameCallback
	mux sync.Mutex
	track *webrtc.Track
	IsActive bool
}

func (ac *AudioCapture) Init() (error) {
	log.Println("Init Audio Capture")
	// gst-launch-1.0 -v pulsesrc ! audioconvert ! audioresample ! audio/x-raw,channels=1,rate=16000 ! filesink location=/dev/stdout | livecaption
	ac.IsActive = false
	
	// samplerate or 16000 * numChannels or 1 * time in ms (60ms or 0.06)  = 960
	// 20ms - the delay added for write intervals
	ac.sampleSize = int(float32(ac.Settings.SampleRate * ac.Settings.Channels) * ac.Settings.Latency)
	log.Println("audioSampleSize", ac.sampleSize)
	// this means the frame size should be calculated back to 60
	ac.buffer = make([]int16, ac.sampleSize)
	ac.inputChannel = make(chan int)
	
	// all the port audio stuff has to be in same goroutine else it no worky
	go func(){
		err := portaudio.Initialize()
		chk(err)
		defer portaudio.Terminate()
		// 960 sample size is 60ms of capture
		in := make([]int16, ac.sampleSize)
		ac.mic, err = portaudio.OpenDefaultStream(ac.Settings.Channels, 0, float64(ac.Settings.SampleRate), ac.sampleSize, in)
		// s
		chk(err)
		//ac.mic = mic
		defer ac.mic.Close()
		chk(ac.mic.Start())
		
		for {
			// calling read populates the buffer passed into the mic stream
			chk(ac.mic.Read())
			// now can send buffer to onFrame
			ac.onFrame(in) //ac.buffer)
			select {
			case <-ac.inputChannel:
				return
			default:
			}
		}
		
		chk(ac.mic.Stop())
		
	}()
	
	return nil
}

func (ac *AudioCapture) Start() error {
	log.Println("Started Capture")
	ac.mux.Lock()
	if !ac.IsActive {
		ac.IsActive = true
	}
	ac.mux.Unlock()

	return nil
}

func (ac *AudioCapture) Stop() error {
	// TODO
	ac.mux.Lock()
	if ac.IsActive {
		ac.IsActive = false
	}
	ac.mux.Unlock()
	close(ac.inputChannel)
	//ac.mic.Stop()
	return nil
}

func (ac *AudioCapture) SetTrack(track *webrtc.Track) {
	ac.mux.Lock()
	ac.track = track
	ac.mux.Unlock()
}

func (ac *AudioCapture) SetEncoder(encoderName string) (error) {
	log.Println("Attempting to set encoder to: ", encoderName)
	switch encoderName {
	case "OPUS":
		ac.opusEnc, err = opus.NewEncoder(ac.Settings.SampleRate, ac.Settings.Channels, opus.AppAudio)
		chk(err)
	}
	return nil
}

func (ac *AudioCapture) encodeFrame(buff []int16, data []byte) (int, error)  {
	// TODO handle different encoders like VP8 and H264
	n, err := ac.opusEnc.Encode(buff, data)
	if err != nil {
		log.Println("encode: ", err)
		return 0, err
	}
	return n, err
}

func (ac *AudioCapture) onFrame(in []int16) {
	//log.Println("calling frame: ", ac.IsActive)
	
	if ac.IsActive {

		frameSize := len(in) // must be interleaved if stereo
		frameSizeMs := ((frameSize / ac.Settings.Channels) * 1000.0) / ac.Settings.SampleRate  //framesize, channels, 1 sec?, sampleRate
		
		//log.Println("framesize, framesizeMS: ", frameSize, frameSizeMs)
		
		switch frameSizeMs {
		case 5, 10, 20, 40, 60:
			// Good.
			//log.Println("framesizems: ", frameSizeMs)
		default:
			log.Println("Illegal frame size: bytes ms", frameSize, frameSizeMs)
			return
		}

		encData := make([]byte, 1000) // size of bytes != sample size int16
		n, err := ac.encodeFrame(in, encData)
		chk(err)
		
		encData = encData[:n] // only the first N bytes are opus data. Just like io.Reader.
		//log.Println("len encoded audio data", len(encData))

		if ac.track == nil {
			log.Println("track was nil...")
			return
		}
		
		//log.Println("writing sample")
		if err := ac.track.WriteSample(media.Sample{Data: encData, Samples: uint32(ac.sampleSize) }); err != nil {
			log.Println("write sample: ", err)
			return
		}
	}
	
}

func NewAudioCapture(settings AudioCaptureSettings)(*AudioCapture, error) {

	ac := &AudioCapture {
		Settings: settings,
	}
	ac.SetEncoder(settings.EncoderName)
	return ac, nil
}


type MediaCapture struct {
	Video *VideoCapture
	Audio *AudioCapture
}

func (mc *MediaCapture)ApplyWebRTCCapabilities(me *webrtc.MediaEngine){
	log.Println("should take the registered codecs and try and sort them out automagically")
}

func (mc *MediaCapture)Start(){
	log.Println("--- Starting MediaCapture ---")
	mc.Audio.Start()
	mc.Video.Start()
}

func (mc *MediaCapture)Stop(){
	log.Println("--- Stopping MediaCapture ---")
	mc.Audio.Stop()
	mc.Video.Stop()
}

func NewMediaCapture(config TelepresentConfig)(*MediaCapture, error){
	log.Println("Creating Media Capture with Default Settings")
	// defaultVideoSettings := &VideoCaptureSettings {
	// 	DeviceID: 0,
	// 	Width: 640,
	// 	Height: 480,
	// 	Framerate: 10,
	// 	EncoderName: "VP8",
	// }
	// defaultAudioSettings := &AudioCaptureSettings {
	// 	SampleRate: 48000,
	// 	Channels: 1,
	// 	Latency: 0.02,
	// 	EncoderName: "opus",
	// }
	var v *VideoCapture
	var a *AudioCapture
	var err error 
	if config.StreamVideo{
		v, err = NewVideoCapture(config.VideoConfig) // defaultVideoSettings)
		chk(err)
		v.Init()
	}
	if config.StreamAudio {
		a, err = NewAudioCapture(config.AudioConfig) //defaultAudioSettings)
		chk(err)
		a.Init()
	}
	log.Println("video capture:",v)
	log.Println("audio capture:",a)

	mc := &MediaCapture{
		Video: v,
		Audio: a,
	}

	return mc, err
}


type SignalServerClient struct {
	Socket gowebsocket.Socket
	wrtcpeer *webrtc.PeerConnection
	answer webrtc.SessionDescription
}

func (ss *SignalServerClient)Connect(){

	ss.Socket.OnConnectError = func(err error, socket gowebsocket.Socket) {
		log.Println("Received connect error - ", err)
		// retry in 3 seconds
		log.Println("Retrying in 3 seconds")
		time.Sleep(time.Millisecond * time.Duration(3000))
		ss.Socket.Connect()
	}

	ss.Socket.Connect()

}

func NewSignalServerClient(config TelepresentConfig)(*SignalServerClient, error){
	ss := &SignalServerClient {}

	ss.Socket = gowebsocket.New(config.SignalServerURI)
  
	ss.Socket.OnConnected = func(socket gowebsocket.Socket) {
		log.Println("Connected to server, logging in...");
		//mapD := map[string]string{"type": "login", "name": connName}
		ps := []string{config.PeerNamePrefix, config.PeerName}
		loginMsg := &LoginStruct{
			Name: strings.Join(ps, "_"),
			Type: "login",
		}
		msgB, _ := json.Marshal(loginMsg)
		socket.SendBinary(msgB)
		//socket.SendText("{'type':'login', 'name':'gopher'}")
	}
  
	ss.Socket.OnTextMessage = func(message string, socket gowebsocket.Socket) {
		//log.Println("Received message - " + message)
		
		//https://gobyexample.com/json
		var resp SignaServerMessage
		err := json.Unmarshal([]byte(message), &resp)
		//err := socket.Conn.ReadJSON(&resp)
		chk(err)

		switch resp.Type {
		case "login":
			if (resp.Success){
				log.Println("Successful LOGIN")
				// wait for offer
			}
		case "offer":
			log.Println("OFFER: ") // + resp.Offer.SDP)

			//should send answer
			ss.wrtcpeer, ss.answer = createWebRTCPeer(config, resp.Offer)
			
			//_ans, _ := json.Marshal(answer)
			//log.Println("Answer to Offer: " + string(_ans))
			// create response
			answerMsg := &AnswerStruct{
				Name: resp.Name,
				Type: "answer",
				Answer: ss.answer,
			}
			msgB, _ := json.Marshal(answerMsg)
			socket.SendBinary(msgB)
		case "candidate":
			log.Println("CANDIDATE: "+ resp.Candidate.Candidate)
			chk(ss.wrtcpeer.AddICECandidate(resp.Candidate))
		//case "answer":
		//	log.Println("ANSWER: ...")
		case "info":
			log.Println("INFO: " + resp.Message)
		case "leave":
			log.Println("LEAVE: " + resp.Message)
			chk(ss.wrtcpeer.Close())
		case "error":
			log.Println("ERROR: " + resp.Message)
			chk(ss.wrtcpeer.Close())
		default:
			log.Println(resp.Type)
		}

	}
  
	// ss.Socket.OnPingReceived = func(data string, socket gowebsocket.Socket) {
	// 	log.Println("Received ping - " + data)
	// }
  
    // ss.Socket.OnPongReceived = func(data string, socket gowebsocket.Socket) {
	// 	log.Println("Received pong - " + data)
	// }

	ss.Socket.OnDisconnected = func(err error, socket gowebsocket.Socket) {
		log.Println("Disconnected from server ")
		chk(ss.wrtcpeer.Close())
	}
	
	return ss, nil
}


const (
	defaultConfigFile = "config.json"
	audioDeviceName = "C922 Pro Stream Webcam: USB Audio (hw:2,0)"
)

var (
	err      error
	videoPayloadType uint8
	audioPayloadType uint8
	dataChannelHandlers map[string]DataChannelHandler
	ssClient *SignalServerClient
	defaultVideoPayloadType uint8 = webrtc.DefaultPayloadTypeVP8  //webrtc.DefaultPayloadTypeH264 / VP8
	defaultAudioPayloadType uint8 = webrtc.DefaultPayloadTypeOpus  //webrtc.DefaultPayloadTypeG711 / Opus
)


func main() {
	log.Println("IN MAIN")
	// proper graceful shutdown handling
	// https://stackoverflow.com/questions/55426907/goroutine-didnt-respect-ctx-done-or-quit-properly
	ctx := context.Background()
    ctx, cancel := context.WithCancel(ctx)
    terminated := monitor(ctx, cancel)
    defer func() {
        cancel()
        log.Println("Cleaned up")
        <-terminated // wait for the monior goroutine quit
    }()

	config := LoadConfiguration(defaultConfigFile)

	ssClient, err := NewSignalServerClient(config)
	chk(err)
	ssClient.Connect()
	

	// block until signal to close is caught
	select {
	case <-ctx.Done():
		ssClient.Socket.Close()
        log.Println("notified to quit")
	}
	
}

func monitor(ctx context.Context, cancel context.CancelFunc) <-chan interface{} {
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt)
    terminated := make(chan interface{})
    go func() {
        defer close(terminated)
        defer log.Println("Stopped monitoring1")
        defer signal.Stop(c)
        select {
        case <-c:
            log.Println("Got interrupt singnal")
            cancel()
        case <-ctx.Done():
        }
    }()
    return terminated
}


func parseGamepadDataChannel(data []byte){
	/*
	Example Message
	{"id":"046d-c21f-Logitech Gamepad F710","state":{"FACE_1":0,"FACE_2":0,"FACE_3":0,"FACE_4":0,
	 "LEFT_TOP_SHOULDER":0,"RIGHT_TOP_SHOULDER":0,"LEFT_BOTTOM_SHOULDER":0,"RIGHT_BOTTOM_SHOULDER":1,
	 "SELECT_BACK":0,"START_FORWARD":0,"LEFT_STICK":0,"RIGHT_STICK":0,"DPAD_UP":0,"DPAD_DOWN":0,
	 "DPAD_LEFT":0,"DPAD_RIGHT":0,"HOME":0,"LEFT_STICK_X":0,"LEFT_STICK_Y":0,"RIGHT_STICK_X":0,"RIGHT_STICK_Y":0},"ts":87090}
	*/
	// parse message, send as ROS message
	//log.Println("Handling Gamepad Data!")
	var gpData GamepadData
	err := json.Unmarshal(data, &gpData)
	chk(err)

	// send to ROS
	/*
	# node.get_logger().info("got gamepad message: {0}".format(gp_state))
	# send over ros2
	msg = Joy()
	# msg.header.timestamp = time.time()
	# we have 2 values for dpad input, but joy msg wants just
	# one value to represent the two, so map to pos/neg
	def map_val(one, two):
		if one > 0:
			return -1
		elif two > 0:
			return 1
		else:
			return 0

	#TODO map these out better...
	msg.axes = [
		float(gp_state['LEFT_STICK_X']),
		float(gp_state['LEFT_STICK_Y']),
		float(gp_state['LEFT_BOTTOM_SHOULDER']),
		float(gp_state['RIGHT_STICK_X']),
		float(gp_state['RIGHT_STICK_Y']),
		float(gp_state['RIGHT_BOTTOM_SHOULDER']),
		float(map_val(gp_state['DPAD_LEFT'], gp_state['DPAD_RIGHT'])),
		float(map_val(gp_state['DPAD_UP'], gp_state['DPAD_DOWN']))
	]
	msg.buttons = [
		int(gp_state['FACE_1']), # a
		int(gp_state['FACE_2']), # b
		int(gp_state['FACE_3']), # x
		int(gp_state['FACE_4']), # y
		int(gp_state['LEFT_TOP_SHOULDER']),
		int(gp_state['RIGHT_TOP_SHOULDER']),
		int(gp_state['SELECT_BACK']),
		int(gp_state['START_FORWARD']), 
		0, # l stick click
		0  # r stick click
	]
	
	try:
		node.joy_pub.publish(msg)
	except Exception as exc:
		node.get_logger().error(str(exc))
	*/

}


//(webrtc.SessionDescription)
func createWebRTCPeer(config TelepresentConfig, offer webrtc.SessionDescription) (*webrtc.PeerConnection, webrtc.SessionDescription){

	log.Println("Creating Peer")

	// We make our own mediaEngine so we can place the sender's codecs in it.  This because we must use the
	// dynamic media type from the sender in our answer. This is not required if we are the offerer
	mediaEngine := webrtc.MediaEngine{}
	err = mediaEngine.PopulateFromSDP(offer)
	chk(err)

	// list of encoders to use by order of best encoding results
	// need to match against what the other peer can do...
	// NOTE: the names might match but it doesn't mean the type value int does...
	//encoderPrefCtr = 0
	//encoderPreferenceList := []string{"VP9", "VP8", "H264"}

	// Setup the codecs you want to use.
	// Only support VP8 and OPUS, this makes our WebM muxer code simpler
	//mediaEngine.RegisterCodec(webrtc.NewRTPVP8Codec(webrtc.DefaultPayloadTypeVP8, 90000))
	//mediaEngine.RegisterCodec(webrtc.NewRTPOpusCodec(webrtc.DefaultPayloadTypeOpus, 48000))

	// Search for VP8 Payload type. If the offer doesn't support VP8 exit since
	// since they won't be able to decode anything we send them
	// TODO, add VP9 and H264 options too
	
	var wantVideo bool = false
	var wantAudio bool = false
	if len(mediaEngine.GetCodecsByKind(webrtc.RTPCodecTypeVideo)) > 0 {
		wantVideo = true
	}
	if len(mediaEngine.GetCodecsByKind(webrtc.RTPCodecTypeAudio)) > 0 {
		wantAudio = true
	}

	
	if config.StreamVideo == true {
		for _, videoCodec := range mediaEngine.GetCodecsByKind(webrtc.RTPCodecTypeVideo) {
			fmt.Println("Video codec available: " + videoCodec.Name +  " type: ", videoCodec.PayloadType, defaultVideoPayloadType)
			if videoCodec.Name == config.VideoConfig.EncoderName && videoCodec.PayloadType == defaultVideoPayloadType {
				videoPayloadType = videoCodec.PayloadType
				break
			}
		}
		if videoPayloadType == 0 {
			panic("Remote peer does not support VP8")
		}
	}
	
	if config.StreamAudio == true {
		for _, audioCodec := range mediaEngine.GetCodecsByKind(webrtc.RTPCodecTypeAudio) {
			fmt.Println("Audio codec available: " + audioCodec.Name + " type: ", audioCodec.PayloadType, defaultAudioPayloadType)
			if audioCodec.Name == config.AudioConfig.EncoderName && audioCodec.PayloadType == defaultAudioPayloadType {
				audioPayloadType = audioCodec.PayloadType
				break
			}
		}
		if audioPayloadType == 0 {
			panic("Remote peer does not support any audio codecs")
		}
	}

	// might be best to create settings now that info from
	// sdp is known about what codecs can be used
	mediaCapture, err := NewMediaCapture(config)
	chk(err)
	log.Println(mediaCapture)

	// Create a new RTCPeerConnection
	// TODO update the URLS to include custom coturn server
	api := webrtc.NewAPI(webrtc.WithMediaEngine(mediaEngine))
	peerConnection, err := api.NewPeerConnection(webrtc.Configuration{
		ICEServers: config.IceServers,
	})
	chk(err)

	// create the tracks
	log.Println("Creating Video and Audio Tracks")
	log.Println(" ===== format types ===== ")
	//ensure both sides want video
	if config.StreamVideo == true && wantVideo {

		log.Println(" === video format types === ")
		log.Println(webrtc.DefaultPayloadTypeH264)
		log.Println(videoPayloadType)

		videoTrack, err := peerConnection.NewTrack(videoPayloadType, rand.Uint32(), "video", config.PeerName+"_video") //videoPayloadType
		chk(err)
		if _, err = peerConnection.AddTrack(videoTrack); err != nil {
			panic(err)
		}

		mediaCapture.Video.SetTrack(videoTrack)
		mediaCapture.Video.Start()
	}
	
	// ensure both sides want audio
	if config.StreamAudio == true && wantAudio{

		log.Println(" === audio format types === ")
		log.Println(webrtc.DefaultPayloadTypeOpus)
		log.Println(audioPayloadType)

		// Create a audio track
		audioTrack, err := peerConnection.NewTrack(audioPayloadType, rand.Uint32(), "audio", config.PeerName+"_audio") //audioPayloadType, webrtc.DefaultPayloadTypeOpus
		chk(err)
		_, err = peerConnection.AddTrack(audioTrack)
		if err != nil {
			panic(err)
		}
		mediaCapture.Audio.SetTrack(audioTrack)
		mediaCapture.Audio.Start()
	}
	log.Println(" ===== end format types ===== ")

	

	// Register data channel creation handling
	peerConnection.OnDataChannel(func(d *webrtc.DataChannel) {
		fmt.Printf("New DataChannel %s %d\n", d.Label(), d.ID())

		// Register channel opening handling
		d.OnOpen(func() {
			fmt.Printf("Data channel '%s'-'%d' open. Random messages will now be sent to any connected DataChannels every 5 seconds\n", d.Label(), d.ID())

			// TODO probably get rid of this?
			for range time.NewTicker(5 * time.Second).C {
				if d.ReadyState() == webrtc.DataChannelStateOpen {
					message := RandSeq(15)
					fmt.Printf("Sending '%s'\n", message)

					// Send the message as text
					sendErr := d.SendText(message)
					chk(sendErr)
				}
			}
		})

		// Register text message handling
		d.OnMessage(func(msg webrtc.DataChannelMessage) {
			//fmt.Printf("Message from DataChannel '%s': '%s'\n", d.Label(), string(msg.Data))

			if d.ReadyState() == webrtc.DataChannelStateOpen {
				if dch, ok := dataChannelHandlers[d.Label()]; ok {
					dch(msg.Data)
				}
			}
		})

		d.OnClose(func(){
			log.Println("data channel closed")
		})

		d.OnError(func(err error){
			log.Println("error with data channel closed:", err)
		})
	})

	// Set the handler for ICE connection state
	// This will notify you when the peer has connected/disconnected
	peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
		log.Println("ICE Connection State has changed: ", connectionState.String())
		switch connectionState {
		case webrtc.ICEConnectionStateConnected:
			log.Println("ICE connected!")
		case webrtc.ICEConnectionStateDisconnected:
			log.Println("ICE disconnected :(")
		case webrtc.ICEConnectionStateFailed:
			log.Println("WTF, the connection failed")
		}
	})

	peerConnection.OnConnectionStateChange(func(connectionState webrtc.PeerConnectionState){
		log.Println("Peer Connection State has changed: ", connectionState.String())
		switch connectionState {
		case webrtc.PeerConnectionStateConnected:
			log.Println("Peer Connected")
			//mediaCapture Start..?
		case webrtc.PeerConnectionStateDisconnected:
			log.Println("Peer Disconnected")
			mediaCapture.Stop()
		case webrtc.PeerConnectionStateFailed:
			log.Println("Peer Connection Failed")
			mediaCapture.Stop()
		case webrtc.PeerConnectionStateClosed:
			log.Println("Peer Connection Failed")
			mediaCapture.Stop()
			//mediaCapture := MediaCapture{}
		}
	})
	

	// Set the remote SessionDescription
	if err = peerConnection.SetRemoteDescription(offer); err != nil {
		panic(err)
	}

	// Create answer
	answer, err := peerConnection.CreateAnswer(nil)
	chk(err)

	// Sets the LocalDescription, and starts our UDP listeners
	if err = peerConnection.SetLocalDescription(answer); err != nil {
		panic(err)
	}
	
	// Output the answer in base64 so we can paste it in browser
	//fmt.Println(signal.Encode(answer))
	return peerConnection, answer

}


//signal

// RandSeq generates a random string to serve as dummy data
func RandSeq(n int) string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[r.Intn(len(letters))]
	}
	return string(b)
}


type TelepresentConfig struct {
	SignalServerURI string `json:"signalServerURI"`
	PeerName 		string `json:"peerName"`
	PeerNamePrefix  string `json:"peerNamePrefix"`
	StreamVideo 	bool `json:"streamVideo"`
	StreamAudio 	bool `json:"streamAudio"`
	VideoConfig VideoCaptureSettings `json:"video"`
	AudioConfig AudioCaptureSettings `json:"audio"`
	IceServers []webrtc.ICEServer `json:"iceServers"`
}

func LoadConfiguration(filename string) TelepresentConfig {
	var config TelepresentConfig
	rootPath, err := os.Getwd()
	chk(err)
	configPath := path.Join(rootPath, filename)
	log.Println("config file full path: ", configPath)

    configFile, err := os.Open(configPath) //ioutil.ReadFile(configPath)
	chk(err)
	defer configFile.Close()
    //if err != nil {
    //    fmt.Println(err.Error())
	//}
	//_ = json.Unmarshal([]byte(configFile), &config)
    jsonParser := json.NewDecoder(configFile)
	jsonParser.Decode(&config)
	log.Println("Config Loaded", config)
	log.Println("derp: ", config.StreamVideo, config.StreamAudio)

	//setup some globals - get rid of this?
	switch config.VideoConfig.EncoderName {
	case "VP8":
		defaultVideoPayloadType = webrtc.DefaultPayloadTypeVP8
	case "VP9":
		defaultVideoPayloadType = webrtc.DefaultPayloadTypeVP9
	case "H264":
		defaultVideoPayloadType = webrtc.DefaultPayloadTypeH264
	default:
		defaultVideoPayloadType = webrtc.DefaultPayloadTypeVP8
	}
	switch config.AudioConfig.EncoderName {
	case "OPUS":
		defaultAudioPayloadType = webrtc.DefaultPayloadTypeOpus
	}

    return config
}


// generic error handling
func chk(err error) {
	if err != nil {
		log.Println(err)
		panic(err)
	}
}