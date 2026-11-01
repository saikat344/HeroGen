"""Frozen Muse Image prompts for repainting renders of our own mesh. A change is a new name, never an edit in place."""

RF2R = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. Keep everything in the first image exactly "
    "where it is: the same framing and size (do not zoom, crop, recentre or rotate), outline, pose, head angle, "
    "proportions, hairstyle shape and colours, and the same plain grey background. Only make it sharp and "
    "clean: crisp eyes (irises, pupils, eyelids, lashes), eyebrows, nose, lips and ears wherever they are "
    "visible; a clean hairline and hair locks; clean clothing edges, seams and folds. The second image is the "
    "character's design: make the details match it, but never copy its framing, pose or viewing angle, and "
    "never add a face or features where the first image does not show them. Use flat, even, shadowless "
    "lighting: do not add shadows, highlights or shading."
)

RF3H = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. The outline is fixed: keep the exact "
    "silhouette, size and framing (do not zoom, crop, recentre or rotate), and keep every region where it is: "
    "hair stays hair, skin stays skin, clothing stays clothing. Give the hair crisp, fine strands and locks "
    "that follow the given shape and flow of the hair, with clean ends; never change its outline, length, "
    "parting or volume, and never add ears, a face or features where the first image shows hair. Sharpen "
    "clothing seams, folds and trims. The second image is the character's design: use it only for colours, "
    "materials and hair texture, never for framing, pose, hairstyle shape or viewing angle. Use flat, even, "
    "shadowless lighting: do not add shadows, highlights or shading."
)

RF4U = (
    "Repaint the first image, a close-up render of a 3D character, with fine surface detail. Keep the exact "
    "outline, geometry, pose, viewpoint, framing and colours of the first image: do not zoom, crop, recentre, "
    "rotate, add objects or change the shape. The body continues outside the frame; do not complete it. "
    "The second image is a matching crop of the character's turnaround artwork. Reproduce its specific "
    "buckle and medallion designs, rivet rows, stitching, chainmail rings, scratches, rust, grime and edge "
    "wear wherever those materials and details occur on the first image. Keep their distinctive colours "
    "and material contrast. The third image is the full character cover for design context only, never "
    "for framing or pose. Skin stays skin and hair stays hair; never add a face where none is visible. "
    "Use flat, even lighting with no cast shadows and no directional lighting."
)

RF5U = (
    "Repaint the first image, a close-up render of a 3D character, with fine surface detail. The first image "
    "is authoritative for the exact geometry, outline, pose, viewpoint, framing, material regions and "
    "colours. Keep every surface and object exactly where it is; do not zoom, crop, recentre, rotate, add "
    "objects or change shapes. A surface cut by the frame continues outside it: never add a hem, border, "
    "panel edge or material transition at a crop edge. Preserve material identity and garment construction: "
    "cloth stays unbroken cloth with the same folds; existing metal, leather, skin and hair keep their "
    "existing regions. Never paint armor plates, buckles, belts, chainmail, openings or patches onto plain "
    "fabric. The second image is a matching crop of the turnaround artwork. Transfer only its specific "
    "details onto the corresponding materials already present in the first image. On existing armor and "
    "leather, reproduce the matching buckle and medallion designs, rivet rows, scratches, rust, grime and "
    "edge wear. On fabric, add fine weave and reproduce only the embroidery and stitching actually shown in "
    "the reference, in their existing positions; do not invent decorative rows, tears, new seams or borders. "
    "Do not copy the second image's object layout, proportions or folds over the first image. The third image "
    "is the full cover for character identity only; never copy objects, materials, pose or framing from it "
    "into this close-up. Never add a face where none is visible. Keep the same background. Use flat, even "
    "illumination with no cast shadows or directional lighting."
)

PROMPTS = {"RF2R": RF2R, "RF3H": RF3H, "RF4U": RF4U, "RF5U": RF5U}

VIEWS = {
    0: "seen from the front",
    40: "seen from a three-quarter front angle",
    45: "seen from a three-quarter front angle",
    90: "seen exactly from the side, in profile",
    135: "seen from behind at three-quarters, so mostly the back and side of the head",
    180: "seen from directly behind, so its face is hidden",
}


def prompt(name, angle):
    text = PROMPTS[name]
    return text.format(view=VIEWS[abs((int(angle) + 180) % 360 - 180)]) if "{view}" in text else text
