-- Optional rig test for a HeroGen R15 rig. Written by HeroGen's Roblox export (stage 6). Not yet verified in a
-- Studio session. Put this Script inside the rig model in Workspace and press Run (F8) or Play (F5).
--
-- It anchors the HumanoidRootPart so the rig stays in place, then plays stock R15 catalog animations (the ids
-- the Creator documentation lists in animation/using: "Catalog animations") and prints, every second, the largest
-- distance between the two RigAttachments of any joint. On a correct rig the pair stays together (< 0.001
-- studs) in every pose; a joint built from the wrong attachments drifts apart.

local model = script.Parent
local humanoid = model:WaitForChild("Humanoid")
local hrp = model:WaitForChild("HumanoidRootPart")
hrp.Anchored = true

local animator = humanoid:FindFirstChildOfClass("Animator") or Instance.new("Animator")
animator.Parent = humanoid

local ANIMS = {
	{ "Ninja run", "rbxassetid://656118852" },
	{ "Ninja idle 1", "rbxassetid://656117400" },
	{ "Ninja idle 2", "rbxassetid://656118341" },
	{ "Ninja jump", "rbxassetid://656117878" },
}

local function jointPairs()
	local pairsList = {}
	for _, j in model:GetDescendants() do
		if j:IsA("Motor6D") then
			local an = j.Name .. "RigAttachment"
			table.insert(pairsList, { j.Name, j.Part0:FindFirstChild(an), j.Part1:FindFirstChild(an), j.ClassName })
		elseif j:IsA("AnimationConstraint") then
			table.insert(pairsList, { j.Name, j.Attachment0, j.Attachment1, j.ClassName })
		end
	end
	return pairsList
end

local joints = jointPairs()
local classes = {}
for _, p in joints do
	classes[p[4]] = (classes[p[4]] or 0) + 1
end
for c, n in classes do
	print(("[rig_test] %d joints of class %s"):format(n, c))
end
assert(#joints == 15, ("[rig_test] expected 15 joints, found %d"):format(#joints))

task.spawn(function()
	while true do
		local worst, worstName = 0, ""
		for _, p in jointPairs() do
			local d = (p[2].WorldPosition - p[3].WorldPosition).Magnitude
			if d > worst then
				worst, worstName = d, p[1]
			end
		end
		print(("[rig_test] worst joint separation %.5f studs (%s)"):format(worst, worstName))
		task.wait(1)
	end
end)

while true do
	for _, a in ANIMS do
		local anim = Instance.new("Animation")
		anim.AnimationId = a[2]
		local track = animator:LoadAnimation(anim)
		track.Looped = true
		track:Play()
		print("[rig_test] playing " .. a[1])
		task.wait(5)
		track:Stop()
	end
end
