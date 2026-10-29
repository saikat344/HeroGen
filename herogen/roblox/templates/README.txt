HeroGen R15 character for Roblox Studio
=======================================

This folder holds your hero as a Roblox R15 rig:

  - {{RIG_FILE}}: the rig, with 15 body parts, the joints and attachments, and HipHeight {{HIP_HEIGHT}}
  - {{BIND_FILE}}: a Studio script that puts the hero's meshes onto the rig
  - {{RIG_TEST_FILE}}: an optional Studio script that plays stock animations on the rig
  - {{RECORD_PATH}}: the checks HeroGen ran on the rig

A Roblox model file can't carry mesh geometry, so the meshes come in through Studio's 3D Importer from the
hero's body file:

  {{GLB_PATH}}

Studio's 3D Importer uploads the meshes and the texture to your Roblox account as new assets. Nothing else in
these steps uploads or publishes anything.

Note: these steps have not yet been verified in a Roblox Studio session.


1. Import the meshes
--------------------
  1. Open your place in Roblox Studio.
  2. Choose File > Import 3D and pick {{GLB_FILE}} (the path above).
  3. In the importer, set:
       Scale Unit: Stud
       World Forward: Front, World Up: Top
       Rig Type: No Rig
  4. Click Import. A model named "{{IMPORTED_NAME}}" appears in Workspace.

2. Insert the rig
-----------------
  1. In the Explorer, right-click Workspace and choose Insert from File...
  2. Pick {{RIG_FILE}} from this folder. A model named "{{RIG_NAME}}" appears in Workspace.
     Its parts have no meshes yet, so it is invisible until step 3.

3. Bind the meshes
------------------
  1. Open {{BIND_FILE}} in a text editor and copy all of it.
  2. In Studio, open View > Command Bar, paste, and press Enter.
  3. The Output window should show:
       fit: residual ... studs        (under 0.01)
       bound 15 meshes; rest-pose joint error ... studs
     If it prints an error instead, the import doesn't match this rig and the rig is left unchanged:
     delete "{{IMPORTED_NAME}}", import {{GLB_FILE}} again with Scale Unit set to Stud, and run the script
     again.
  4. Delete the "{{IMPORTED_NAME}}" model; the rig now holds the meshes.
  5. Check the result: select {{RIG_NAME}} > Humanoid; RigType is R15 and HipHeight is about {{HIP_HEIGHT}}.
     Seen from the front, the hero's left arm is on the right of the screen.

4. Optional: test the joints
----------------------------
  1. Insert a Script into {{RIG_NAME}} and replace its text with the text of {{RIG_TEST_FILE}}.
  2. Press Run (F8). The Output window should show "15 joints of class Motor6D", the rig should play the run,
     idle and jump animations with its joints bending the right way, and every "worst joint separation" line
     should stay under 0.001 studs.
  3. Stop, then delete the Script.

5. Play as the hero
-------------------
  1. Move {{RIG_NAME}} into StarterPlayer and rename it StarterCharacter.
  2. Select StarterPlayer and turn off LoadCharacterAppearance.
  3. Press Play. The hero should stand with its feet on the floor and idle, walk and jump with Roblox's stock
     R15 animations.
  4. If the hero moves without animating, the character has no Animate script. Remove StarterCharacter, press
     Play, copy the Animate script from your character in Workspace, stop, put StarterCharacter back, and paste
     the Animate script into it.

To reuse the finished rig in other places, right-click it and choose Save to File... (or Save to Roblox...).
