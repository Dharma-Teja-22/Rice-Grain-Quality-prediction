#RICE_QUALITY_DETECTION_USING_IMAGE_PROCESSING AND DATA VIZUALIZATION
#IMPORTING DATASETS
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly import subplots
import pandas as pd
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output, State
import cv2
import PIL.Image as image
from io import BytesIO
import base64
from matplotlib import pyplot as plt
app = dash.Dash(__name__)
#project explanation
text1="""  Basmati rice"""
text2="""  It is high low glycemic in dietary fiber and index. It is also low-carb and gluten-free. It is used for brain health, weight loss, digestive health, diabetic management."""
text3="""  Sona Masuvi """
text4="""  It has zero fat and low sodium. It helps for health issues like blood pressure and cholestrol levels. It has high quality carbs and protefos. """
text5="""  Brown rice """
text6="""  It has nutritimal superiority makes us healthier. It helps in weight loss. It controls blood pressure sugar level, adds digestion and is neuropro- -tective effective."""
text7="""  Mogra rice """
text8="""  It is low fat and gluten free, essential in amino acids and folic acids. It helps in digestion free problems."""
text9="""  White rice """
text10=""" It gives  us energy to function, case  to digest. It is full of nutrients like manganese, iron, and B vitamins. It Supports bones, nerves, muscles."""
text11=""" Sambar Rice"""
text12=""" It helps Immunity, stamina Strength- in improving digestion, muscle and nerve. It also has a low glycemic which makes it good for index, diabetics. It is a rich source of carbohydrates, dietary fiber and essential minerals like magnesium, potassium & zinc."""
text13=""" Indrayani rice """
text14=""" It 86 used for supper.It supports digestion, boosting metabolism, promoting weight loss etc. It delays the ageing process. It helps levels to stabilize and strengthen blood sugar Immunity. """
#classification of rice particals
def get_classification(ratio):
    ratio =round(ratio,1)
    toret=""
    if(ratio>=3 and ratio<3.5):
        toret="Slender"
    elif(ratio>=2.1 and ratio<3):
        toret="Medium"
    elif(ratio>=1.1 and ratio<2.1):
        toret="Bold"
    elif(ratio>0.9 and ratio<=1):
        toret="Round"
    else:
        toret="Dust"
    return toret

#initialisig the values
classification = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}
avg = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}

#load in greyscale mode
from IPython.display import display, Image
img = cv2.imread("rice.png",0)
display(Image(filename='rice.png'))

#histogram part of the image
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('rice.png',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

#convert into binary
# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((5,5),np.float32)/9
dst = cv2.filter2D(binary,-1,kernel)
# -1 : depth of the destination image
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion
erosion = cv2.erode(dst,kernel2,iterations = 1)

#dilation
dilation = cv2.dilate(erosion,kernel2,iterations = 1)

#edge detection
edges = cv2.Canny(dilation,100,200)

#size detection
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("No. of rice grains=",len(contours))
total_ar=0

#counting impurities
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if(aspect_ratio<1):
        aspect_ratio=1/aspect_ratio
    #print(round(aspect_ratio,2),get_classification(aspect_ratio))
    classification[get_classification(aspect_ratio)] += 1
    if get_classification(aspect_ratio) != "Dust":
        total_ar+=aspect_ratio
    if get_classification(aspect_ratio) != "Dust":
        avg[get_classification(aspect_ratio)] += aspect_ratio

#getting the average value
avg_ar=total_ar/len(contours)

#setting the values for classification of rice
if classification['Slender']!=0:
    avg['Slender'] = avg['Slender']/classification['Slender']
if classification['Medium']!=0:
    avg['Medium'] = avg['Medium']/classification['Medium']
if classification['Bold']!=0:
    avg['Bold'] = avg['Bold']/classification['Bold']
if classification['Round']!=0:
    avg['Round'] = avg['Round']/classification['Round']

#saving different types of images
cv2.imwrite("img.jpg", img)
cv2.imwrite("binary.jpg", binary)
cv2.imwrite("dst.jpg", dst)
cv2.imwrite("erosion.jpg", erosion)
cv2.imwrite("dilation.jpg", dilation)
cv2.imwrite("edges.jpg", edges)

#histogram part for the edge part of the image
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('edges.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

#converting rgb to bgr
def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

#updating the image
def update_image(pic):
    img = readb64(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classification1 = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}
    avg1 = {"Slender":0, "Medium":0, "Bold":0, "Round":0, "Dust":0}
    #convert into binary
    ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
    #averaging filter
    kernel = np.ones((5,5),np.float32)/9
    dst = cv2.filter2D(binary,-1,kernel)# -1 : depth of the destination image

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    #erosion
    erosion = cv2.erode(dst,kernel2,iterations = 1)

    #dilation
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)

    #edge detection
    edges = cv2.Canny(dilation,100,200)

    ### Size detection
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("No. of rice grains=",len(contours))
    total_ar1=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if(aspect_ratio<1):
            aspect_ratio=1/aspect_ratio
        print(round(aspect_ratio,2),get_classification(aspect_ratio))
        classification1[get_classification(aspect_ratio)] += 1
        if get_classification(aspect_ratio) != "Dust":
            total_ar1+=aspect_ratio
        if get_classification(aspect_ratio) != "Dust":
            avg1[get_classification(aspect_ratio)] += aspect_ratio
    avg_ar1=total_ar1/len(contours)
    if classification1['Slender']!=0:
        avg1['Slender'] = avg1['Slender']/classification1['Slender']
    if classification1['Medium']!=0:
        avg1['Medium'] = avg1['Medium']/classification1['Medium']
    if classification1['Bold']!=0:
        avg1['Bold'] = avg1['Bold']/classification1['Bold']
    if classification1['Round']!=0:
        avg1['Round'] = avg1['Round']/classification1['Round']
    cv2.imwrite("img1.jpg", img)
    cv2.imwrite("binary1.jpg", binary)
    cv2.imwrite("dst1.jpg", dst)
    cv2.imwrite("erosion1.jpg", erosion)
    cv2.imwrite("dilation1.jpg", dilation)
    cv2.imwrite("edges1.jpg", edges)
    return classification1,avg1,avg_ar1


#displaying image
def get_image(path):
    img=image.open(path)
    # Constants
    img_width = 710
    img_height = 550
    scale_factor = 0.5
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        scaleanchor="x"
    )
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    fig.show(config={'doubleClick': 'reset'})
    return fig

#average aspect vs classification plot
def get_plot1(classification = classification, avg = avg, avg_ar = avg_ar):
    fig = subplots.make_subplots(rows=1,cols=1,specs=[[{"type":"bar"}]], shared_xaxes=True)
    print(list(classification.keys()))
    print(list(classification.values()))
    plot1 = go.Bar(x=list(classification.keys()), y=list(classification.values()), name="Particles")
    plot2 = go.Bar(x=list(avg.keys()), y=list(avg.values()), name="Avg. Aspect Ratio")
    fig.add_trace(plot1,1,1)
    fig.add_trace(plot2,1,1)
    fig.add_shape(
        type="line",
        x0=0,
        y0=round(avg_ar,2),
        x1=5,
        y1=round(avg_ar,2),
        line=dict(
            color="LightSeaGreen",
            width=4,
            dash="dashdot",
        ),
    )
    fig.update_layout(
        width = 600,
        height = 350,
        margin = {"l": 5, "r": 5, "t": 30, "b": 5},
        title = "Average Aspect Ratio Vs Classification",
        template = "plotly_dark"
    )
    return fig

#quality annalysis
def get_plot2(classification = classification):
    fig = subplots.make_subplots(rows=1,cols=1,specs=[[{"type":"pie"}]])
    rice = sum(list(classification.values())) - classification['Dust']
    dust = classification['Dust']
    values = [rice, dust]
    labels = ["Rice", "Dust"]
    plot1 = go.Pie(labels=labels, values=values, hole=.3)
    fig.add_trace(plot1,1,1)
    fig.update_layout(
        width = 600,
        height = 350,
        margin = {"l": 65, "r": 5, "t": 60, "b": 50},
        title = "Quality Analysis",
        template = "plotly_dark"
    )
    return fig

#
app.layout = html.Div([
	html.Div([
		html.Div([
			
		],style={"float":"left","padding" : "5px 0 5px 50px"}),
		html.Div(
			children="Rice Grain Quality Analysis and Classification using Machine Learning",
			style={"float":"left","padding" : "10px 0 10px 10px","font-size": "17px", "font-weight" :"600"}
		),
		html.Div([
			html.Div([html.A("Home",href="#home")], style={"float":"left","padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
			html.Div([html.A("Benefits Of Rice Grains",href="#about-project")], style={"float":"left","padding":"0 10px 0 10px","align-items": "center","font-size": "15px", "font-weight" :"600"}),
                                            		],style={"float":"right", "padding": "10px 50px 10px 0px"})
	],className="nav"),
	html.Div([],style={"height":"50px"},id="home"),
	html.Div([
		html.H1(children="Visualisation of Results", style={"text-align":"center", "margin":"0", "padding-bottom" : "20px", "color" : "violet"}),
		html.Div([
			html.Div([
				dcc.Graph(figure=get_plot1(),id="graph1"),
				html.P("Original Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(figure=get_plot2(),id="graph2"),
				html.P("Binary Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
		], style = {"display": "flex", "justify-content": "center", "align-items": "center", "text-align":"center"}),
		html.Div([]),
		html.Div([
			html.Div([
				dcc.Upload([
					'Drag and Drop or ',
					html.A('Select a File')
				],
				style={
					'width': '100%',
					'height': '60px',
					'lineHeight': '60px',
					'borderWidth': '1px',
					'borderStyle': 'dashed',
					'borderRadius': '5px',
					'textAlign': 'center'
				}, id="upload-image"),
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
		], style = {"display": "flex", "justify-content": "center", "align-items": "center", "text-align":"center", "width" : "100%"})
	],style = {"color":"black", "padding" : "20px 0 20px 0", "color" : "whitesmoke"},id='plots'),
	html.Div([
		html.H1(children="Images", style={"text-align":"center", "margin":"0", "padding-bottom" : "20px"}),
		html.Div([
			html.Div([
				dcc.Graph(figure=get_image("img.jpg"),id="img"),
				html.P("Original Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(figure=get_image("binary.jpg"),id="binary"),
				html.P("Binary Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(figure=get_image("dst.jpg"),id="dst"),
				html.P("Dust Image", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"})
		], style = {"display": "flex", "justify-content": "center", "align-items": "center", "text-align":"center"}),
		html.Div([]),
		html.Div([
			html.Div([
				dcc.Graph(figure=get_image("erosion.jpg"),id="erosion"),
				html.P("Erosion", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(figure=get_image("dilation.jpg"),id="dilation"),
				html.P("Dilation", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"}),
			html.Div([
				dcc.Graph(figure=get_image("edges.jpg"),id="edges"),
				html.P("Edge Detection", style={"margin":"0","padding-bottom":"10px"})
			], style = {"display": "block", "justify-content": "center", "align-items": "center", "padding":"0 20px 0 20px"})
		], style = {"display": "flex", "justify-content": "center", "align-items": "center", "text-align":"center"})
	],style = {"color":"black", "background-color" : "lightsteelblue", "border-radius":"40px 40px 40px 40px", "padding" : "20px 0 20px 0"},id='images'),
	html.Div([
		html.H1(children="Benefits Of Rice Grains", style={"text-align":"center", "color":"black"}),
		html.P(children=text1 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text2 ,style={"color":"black", "padding-left":"40px"}),
		html.P(children=text3 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text4 ,style={"color":"black", "padding-left":"40px"}),
		html.P(children=text5 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text6 ,style={"color":"black", "padding-left":"40px"}),
        html.P(children=text7 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text8 ,style={"color":"black", "padding-left":"40px"}),
		html.P(children=text9 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text10 ,style={"color":"black", "padding-left":"40px"}),
		html.P(children=text11 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text12 ,style={"color":"black", "padding-left":"40px"}),
        html.P(children=text13 ,style={"color":"black", "font-weight":"bold"}),
		html.P(children=text14 ,style={"color":"black", "padding-left":"40px"}),
        
	],style = {"color":"Black", "padding":"10px 50px 10px 50px"},id="about-project")
])

#app callback
@app.callback([Output('img', 'figure'),
			   Output('binary', 'figure'),
			   Output('dst', 'figure'),
			   Output('erosion', 'figure'),
			   Output('dilation', 'figure'),
			   Output('edges', 'figure'),
			   Output('graph1', 'figure'),
			   Output('graph2', 'figure')],
			  [Input('upload-image', 'contents')])

#updating the outputs
def update_output(list_of_contents):
	if list_of_contents is not None:
		ind = str(list_of_contents).find(",")
		cla,av,av_ar = update_image(list_of_contents[ind:])
		return get_image("img1.jpg"), get_image("binary1.jpg"), get_image("dst1.jpg"), get_image("erosion1.jpg"), get_image("dilation1.jpg"), get_image("edges1.jpg"), get_plot1(cla, av, av_ar), get_plot2(cla)
	else:
		return get_image("img.jpg"), get_image("binary.jpg"), get_image("dst.jpg"), get_image("erosion.jpg"), get_image("dilation.jpg"), get_image("edges.jpg"), get_plot1(), get_plot2()

#hosting the website
if __name__ == '__main__':
	app.run_server(debug=False)
