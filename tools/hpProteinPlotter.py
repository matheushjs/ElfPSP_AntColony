import sys
import json
import vtk

colors = vtk.vtkNamedColors()

class Point:
    def __init__(self, x=0, y=0, z=0, t="H"):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __str__(self):
        return "(" + ", ".join([str(i) for i in [self.x, self.y, self.z, self.t]]) + ")"

def getPoints(filename):
    with open(filename) as fp:
        data = json.load(fp)

    points = []

    coords = data["coords"].strip().split(" ")
    hp = list(data["hpchain"])

    for coord in coords:
        nums = coord.strip("()").split(",")
        nums = [4*int(i) for i in nums]
        points.append(Point(nums[0], nums[1], nums[2]))

    for p, h in zip(points, hp):
        p.t = h

    return points

def sphereActor(x, y, z, color="blue"):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(x, y, z)
    sphereSource.SetRadius(0.8)
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())
    mapper.SetScalarVisibility(False)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(1)
    actor.GetProperty().SetSpecular(0.7)
    actor.GetProperty().SetSpecularPower(20)

    return actor

def lineActor(p1, p2):
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(p1.x, p1.y, p1.z)
    lineSource.SetPoint2(p2.x, p2.y, p2.z)

    tubeSource = vtk.vtkTubeFilter()
    tubeSource.SetInputConnection(lineSource.GetOutputPort())
    tubeSource.SetRadius(1/3)
    tubeSource.SetNumberOfSides(100)
    tubeSource.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("darkslategray"))
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(1)
    actor.GetProperty().SetSpecular(1)
    actor.GetProperty().SetSpecularPower(30)

    return actor

def makeCameraPass():
    # Set crazy shadowing. I do not know what these things do.
    opaque = vtk.vtkOpaquePass()
    peeling = vtk.vtkDepthPeelingPass()
    peeling.SetMaximumNumberOfPeels(200)
    peeling.SetOcclusionRatio(0.1)
    translucent = vtk.vtkTranslucentPass()
    peeling.SetTranslucentPass(translucent)
    volume = vtk.vtkVolumetricPass()
    overlay = vtk.vtkOverlayPass()
    lights = vtk.vtkLightsPass()
    shadowsBaker = vtk.vtkShadowMapBakerPass()
    shadowsBaker.SetResolution(1024)
    shadows = vtk.vtkShadowMapPass()
    shadows.SetShadowMapBakerPass(shadowsBaker)
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadowsBaker)
    passes.AddItem(shadows)
    passes.AddItem(lights)
    passes.AddItem(peeling)
    passes.AddItem(volume)
    passes.AddItem(overlay)
    seq = vtk.vtkSequencePass()
    seq.SetPasses(passes)
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    return cameraP

def addLights(renderer, points):
    # Get some edge coordinates
    minX = min([p.x for p in points])
    maxX = max([p.x for p in points])
    maxY = max([p.y for p in points])

    # Add some lights
    l1 = vtk.vtkLight()
    l1.SetPosition(minX-10,0,0)
    l1.SetFocalPoint(0,0,0)
    l1.SetColor(1,1,1)
    l1.SetIntensity(0.5)
    renderer.AddLight(l1)
    l2 = vtk.vtkLight()
    l2.SetPosition(maxX+10,maxY+10,0)
    l2.SetFocalPoint(0,0,0)
    l2.SetColor(1,1,1)
    l2.SetIntensity(0.5)
    renderer.AddLight(l2)

    # Add something to represent the light source, if you want.
    p = l1.GetPosition()
    renderer.AddActor(sphereActor(p[0], p[1], p[2], "gold"))
    p = l2.GetPosition()
    renderer.AddActor(sphereActor(p[0], p[1], p[2], "gold"))


def main():
    if len(sys.argv) == 1:
        print("Usage: {} [input file]\n".format(sys.argv[0]))
        return 1
    else:
        points = getPoints(sys.argv[1])

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Sphere")
    renderWindow.SetSize(1000, 800)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.SetBackground(colors.GetColor3d("Gainsboro"))
    renderer.SetPass(makeCameraPass())

    addLights(renderer, points)

    # Render first bead
    p = points[0]
    renderer.AddActor(sphereActor(p.x, p.y, p.z, "cyan" if p.t is "P" else "orange"))

    # Render beads
    for p in points[1:]:
        renderer.AddActor(sphereActor(p.x, p.y, p.z, "SteelBlue" if p.t is "P" else "Crimson"))

    # Render bonds
    for i in range(1, len(points)):
        renderer.AddActor(lineActor(points[i-1], points[i]))

    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == '__main__':
    main()
