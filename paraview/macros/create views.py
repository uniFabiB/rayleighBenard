# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
#paraview.simple._DisableFirstRenderCameraReset()


glyphScale = 0.001



names = []
resources = []
items = GetSources().items()


for (k, resource) in items:
    name = k[0]
    names.append(name)
    resources.append(resource)
#    id = k[1]
#    ids.append(id)

n = 0
pvdIndices = []
for i in range(len(names)):
    print(names[i])
    if ".pvd" in names[i]:
        pvdIndices.append(i)

n = len(pvdIndices)
print("number of items", n)
#print("items",items)
fraction = 1.0/n

layout = GetLayoutByName("Layout #1")

### create "splittings" -> windows for me
windowIds = []
for i in range(n):
    if i == 0:
        id = 0
        layout.SplitHorizontal(id, 1.0/(n-i))
    else:
        id = 2**(i+1)-2
        layout.SplitHorizontal(id, 1.0/(n-i))
    windowIds.append(0)


def showArray(source, functionname, view, scaleMin, scaleMax):
    # get display properties
    display = GetDisplayProperties(source, view=view)

    # set scalar coloring
    ColorBy(display, ('POINTS', functionname))

    # rescale color and/or opacity maps used to include current data range
    display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    display.SetScalarBarVisibility(view, True)

    # get color transfer function/color map for 'theta'
    thetaLUT = GetColorTransferFunction(functionname)

    # apply my color scale
    thetaLUT.ApplyPreset('my cool to warm', True)
    
# Rescale transfer function
    thetaLUT.RescaleTransferFunction(scaleMin, scaleMax)

    # disable light kit
    renderView.UseLight = 0

    # Properties modified on renderView1
    renderView.OrientationAxesVisibility = 0


def showGlyph(source, view, scaleFactor):

    # create a new 'Glyph'
    glyph = Glyph(registrationName='Glyph2', Input=source, GlyphType='2D Glyph')
    glyph.OrientationArray = ['POINTS', 'u']
    glyph.ScaleArray = ['POINTS', 'u']
    glyph.ScaleFactor = scaleFactor
    glyph.GlyphTransform = 'Transform2'

    Show(glyph, view)

for i in range(n):
    windowId = windowIds[i]
    sourceIndex = pvdIndices[i]
    resource = resources[sourceIndex]

    # create renderview
    renderView = CreateView('RenderView')

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=renderView, layout=layout, hint=windowId)


    # set active source
    SetActiveSource(resource)

    # show data in view
    testpvdDisplay = Show(resource,renderView)
    renderView.InteractionMode = '2D'

    renderView.ResetCamera(True)
    showArray(resource, "theta", renderView, -1.0, 1.0)
    showGlyph(resource, renderView, glyphScale)



print("done")

# resize frame
#layout1.SetSplitFraction(0, 0.2683790965456156)

# resize frame
#layout1.SetSplitFraction(2, 0.33110976349302607)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------------------
# uncomment the following to render all views
RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
