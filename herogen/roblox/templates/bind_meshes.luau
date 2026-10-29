-- Bind the meshes Studio's 3D Importer made from {{GLB_FILE}} onto the HeroGen R15 rig {{RIG_NAME}}.
-- Written by HeroGen's Roblox export (stage 6). Not yet verified in a Studio session.
-- Paste this whole file into the Studio Command Bar (View > Command Bar) and press Enter.
--
-- Before running, Workspace holds:
--   {{RIG_NAME}}  the rig: in the Explorer, right-click Workspace > Insert from File... > {{RIG_FILE}}
--   {{IMPORTED_NAME}}  the imported meshes: File > Import 3D > {{GLB_FILE}} (Scale Unit: Stud, Rig Type: No Rig)
-- Every section keeps its rig instance (attachments, Motor6D, scale value); ApplyMesh copies the mesh, texture
-- and collision geometry from the imported MeshPart (create.roblox.com/docs/reference/engine/classes/MeshPart).
-- The imported layout is fitted to the rig with one of the 24 axis-aligned rotations plus a translation, so it
-- does not matter whether the Importer baked its axis conversion into the vertices or into the part CFrames.

local RIG_NAME = "{{RIG_NAME}}"
local IMPORTED_NAME = "{{IMPORTED_NAME}}" -- the Importer names the model after the file; change it if renamed

local rig = workspace:FindFirstChild(RIG_NAME)
local src = workspace:FindFirstChild(IMPORTED_NAME)
assert(rig and rig:IsA("Model"), "no rig model named " .. RIG_NAME .. " in Workspace")
assert(src, "no imported model named " .. IMPORTED_NAME .. " in Workspace")

local SECTIONS = {
	"Head", "UpperTorso", "LowerTorso", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm",
	"RightLowerArm", "RightHand", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot",
}

local function baseName(n)
	for _, suffix in { "_Node", "_Geo" } do
		if n:sub(-#suffix) == suffix then
			return n:sub(1, #n - #suffix)
		end
	end
	return n
end

local found = {}
for _, d in src:GetDescendants() do
	if d:IsA("MeshPart") then
		for _, n in { baseName(d.Name), d.Parent and baseName(d.Parent.Name) or "" } do
			if table.find(SECTIONS, n) then
				assert(found[n] == nil or found[n] == d, "two imported MeshParts for " .. n)
				found[n] = d
			end
		end
	end
end
for _, n in SECTIONS do
	assert(found[n], "imported model has no MeshPart for " .. n)
	assert(rig:FindFirstChild(n), "rig has no part " .. n)
end

-- fit: R * importedCentre + t = rigCentre
local rotations = {}
for _, x in { Vector3.xAxis, -Vector3.xAxis, Vector3.yAxis, -Vector3.yAxis, Vector3.zAxis, -Vector3.zAxis } do
	for _, y in { Vector3.xAxis, -Vector3.xAxis, Vector3.yAxis, -Vector3.yAxis, Vector3.zAxis, -Vector3.zAxis } do
		if math.abs(x:Dot(y)) < 0.5 then
			table.insert(rotations, CFrame.fromMatrix(Vector3.zero, x, y))
		end
	end
end
local bestR, bestT, bestErr = nil, nil, math.huge
for _, R in rotations do
	local sum = Vector3.zero
	for _, n in SECTIONS do
		sum += rig[n].Position - R * found[n].Position
	end
	local t = sum / #SECTIONS
	local worst = 0
	for _, n in SECTIONS do
		worst = math.max(worst, (R * found[n].Position + t - rig[n].Position).Magnitude)
	end
	if worst < bestErr then
		bestR, bestT, bestErr = R, t, worst
	end
end
assert(bestErr < 0.01, string.format("imported layout does not match the rig (best fit %.4f studs): wrong Scale Unit or another body?", bestErr))
local fit = CFrame.new(bestT) * bestR
print(string.format("fit: residual %.5f studs, rotation %s", bestErr, tostring(bestR)))

-- Check every section's size before changing anything, so a mismatch leaves the rig as it was.
for _, n in SECTIONS do
	local ext = (fit * found[n].CFrame):VectorToWorldSpace(found[n].Size)
	local d = Vector3.new(math.abs(ext.X), math.abs(ext.Y), math.abs(ext.Z)) - rig[n].Size
	assert(math.max(math.abs(d.X), math.abs(d.Y), math.abs(d.Z)) < 0.01, n .. ": imported size differs from the rig's by " .. tostring(d) .. " studs (wrong Scale Unit or another body?)")
end

for _, n in SECTIONS do
	local part, from = rig[n], found[n]
	local worldAtt = {}
	for _, a in part:GetChildren() do
		if a:IsA("Attachment") then
			worldAtt[a] = part.CFrame * a.CFrame
		end
	end
	part:ApplyMesh(from)
	part.Size = from.Size
	part.CFrame = fit * from.CFrame
	pcall(function()
		part.DoubleSided = from.DoubleSided
	end)
	local sa = from:FindFirstChildOfClass("SurfaceAppearance")
	if sa then
		sa:Clone().Parent = part
	end
	for a, w in worldAtt do
		a.CFrame = part.CFrame:Inverse() * w
	end
end

-- joints follow the moved attachments
local worst = 0
for _, m in rig:GetDescendants() do
	if m:IsA("Motor6D") then
		local an = m.Name .. "RigAttachment"
		local a0, a1 = m.Part0:FindFirstChild(an), m.Part1:FindFirstChild(an)
		m.C0, m.C1 = a0.CFrame, a1.CFrame
		worst = math.max(worst, ((m.Part0.CFrame * m.C0 * m.C1:Inverse()).Position - m.Part1.Position).Magnitude)
	end
end
print(string.format("bound 15 meshes; rest-pose joint error %.6f studs. You can delete '%s' now.", worst, IMPORTED_NAME))
