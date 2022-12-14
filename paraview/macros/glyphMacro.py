# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
testpvd = FindSource('test.pvd')

# create a new 'Glyph'
glyph2 = Glyph(registrationName='Glyph2', Input=testpvd,
    GlyphType='Arrow')
glyph2.OrientationArray = ['POINTS', 'u']
glyph2.ScaleArray = ['POINTS', 'p']
glyph2.ScaleFactor = 0.10400000000000001
glyph2.GlyphTransform = 'Transform2'

# Properties modified on glyph2
glyph2.GlyphType = '2D Glyph'
glyph2.ScaleArray = ['POINTS', 'u']
glyph2.ScaleFactor = 2e-05

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# trace defaults for the display properties.
glyph2Display.Representation = 'Surface'
glyph2Display.ColorArrayName = ['POINTS', 'p']
glyph2Display.LookupTable = pLUT
glyph2Display.SelectTCoordArray = 'None'
glyph2Display.SelectNormalArray = 'None'
glyph2Display.SelectTangentArray = 'None'
glyph2Display.OSPRayScaleArray = 'p'
glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph2Display.SelectOrientationVectors = 'u'
glyph2Display.ScaleFactor = 0.10361053552478552
glyph2Display.SelectScaleArray = 'p'
glyph2Display.GlyphType = 'Arrow'
glyph2Display.GlyphTableIndexArray = 'p'
glyph2Display.GaussianRadius = 0.005180526776239276
glyph2Display.SetScaleArray = ['POINTS', 'p']
glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph2Display.OpacityArray = ['POINTS', 'p']
glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph2Display.DataAxesGrid = 'GridAxesRepresentation'
glyph2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph2Display.ScaleTransferFunction.Points = [-211020.82018987054, 0.0, 0.5, 0.0, 383329.1701538501, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph2Display.OpacityTransferFunction.Points = [-211020.82018987054, 0.0, 0.5, 0.0, 383329.1701538501, 1.0, 0.5, 0.0]

# show color bar/color legend
glyph2Display.SetScalarBarVisibility(renderView1, True)

# find source
testpvd_1 = FindSource('test.pvd')

# find source
testpvd_2 = FindSource('test.pvd')

# find source
glyph1 = FindSource('Glyph1')

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2252, 1220)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.49999999068677425, 4.342671338528846]
renderView1.CameraFocalPoint = [0.5, 0.49999999068677425, 0.0]
renderView1.CameraParallelScale = 1.1474482460286748

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
