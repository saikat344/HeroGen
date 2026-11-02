"""Frozen Muse Image prompts for repainting renders of our own mesh. A change is a new name, never an edit in place."""

from herogen.pipeline.redo import clean_note

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

RF2W = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. Keep everything in the first image exactly "
    "where it is: the same framing and size (do not zoom, crop, recentre or rotate), outline, pose, head angle, "
    "proportions, hairstyle shape and colours, and the same plain grey background. Match the colours, value "
    "range, material contrast and wear of the second image: scratches, pitting, rust, grime, dark crevices "
    "between plates and under straps, and painted edge highlights as in the artwork. Keep crisp eyes (irises, "
    "pupils, eyelids, lashes), eyebrows, nose, lips and ears wherever they are visible; a clean hairline and "
    "hair locks; clean clothing edges, seams and folds. The second image is the character's design: make the "
    "details match it, but never copy its framing, pose or viewing angle, and never add a face or features "
    "where the first image does not show them. Use flat, even lighting with no cast shadows and no directional "
    "lighting that would differ between views; keep only the shading, highlights and wear painted in the artwork."
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

RF5B = (
    "Repaint the first image, a render of our 3D character {view}, as a polished stylised game-character "
    "beauty render. Keep its exact camera, silhouette, pose, framing, proportions, hairstyle shape and all "
    "region boundaries: hair stays hair, skin stays skin, clothing stays clothing. Never add a face where "
    "none is visible. The second image is the nearest beauty reference for this camera yaw (a head crop "
    "for a head close-up): match its character, costume details, colours, crisp edges and rendering style. "
    "Ornaments, jewels, bosses and buckles must match the reference artwork in number and material, and "
    "nothing may be added. "
    "Bake soft studio light into the paint, fixed ABOVE AND IN FRONT OF THE CHARACTER at front yaw zero "
    "in character space, with gentle fill in every view; never rotate the key light to follow the camera. "
    "Keep overlapping surfaces identically shaded across views. Darken occluded creases under the cape, "
    "between straps and inside boot cuffs. Separate leather grain and seams, cloth weave and folds, "
    "restrained metal highlights and softly shaded skin. Sharpen eyes, brows, lips, hair strands, trims "
    "and material edges wherever visible, without moving them. Preserve the first image's uniform grey "
    "background exactly. No rim light, cast ground shadow, scenery, text or added props."
)

# Exposed faces must be reconstructed from the design even when the mesh paint is soft.
RF6B = RF5B.replace(
    "Never add a face where none is visible.",
    "Where exposed facial skin is visible, restore a complete face from the second image: "
    "distinct eyes with irises and pupils, eyelids, eyebrows, nose and lips. "
    "The first image may have blurred or missing facial detail; do not copy that blur, erase "
    "features or leave blank skin in their place. Keep their anatomical positions and the "
    "reference character's identity and expression. Never paint facial features on hair, "
    "a helmet, a covered face or the back of the head.")

RF17B = (
    "Repaint only the visible surfaces of the first image, a render of our 3D character {view}. "
    "The first image alone fixes every pixel of the silhouette, pose, proportions, camera, framing, "
    "garment boundaries and background. Do not rebuild the character or copy the reference's anatomy, "
    "limb widths, folds, pose or object layout. Keep the first image's exact size and grey background. "
    "The last image is a surface design reference fitted to this silhouette, not replacement geometry. "
    "Transfer its crisp material detail, costume markings, embroidery, rivets, seams, straps and wear "
    "to their corresponding existing surfaces, keeping its cloth and leather colours. Never add straps, "
    "belts, buckles or ornaments that the reference does not show. Keep hair, skin, cloth, leather and metal "
    "in their original regions. Keep exposed faces in place; never uncover a helmet or paint a face "
    "on hair or a rear head. Use restrained soft shading with the light fixed above and in front of "
    "the character at front yaw zero, with gentle fill in every view. Sharpen the surface paint "
    "without changing any shape, crop edge or silhouette. No scenery, text, added props or ground shadow."
)

RF6R = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. Keep everything in the first image exactly "
    "where it is: the same framing and size (do not zoom, crop, recentre or rotate), outline, pose, head angle, "
    "proportions, hairstyle shape and colours, and the same plain grey background. Only make it sharp and "
    "clean: crisp eyes (irises, pupils, eyelids, lashes), eyebrows, nose, lips and ears wherever they are "
    "visible; a clean hairline and hair locks; clean clothing edges, seams and folds. The images between the "
    "first and the last are the character's turnaround artwork for this viewpoint: every panel, strap, seam, "
    "trim, buckle, pattern and colour area they show on the surfaces visible here MUST be painted in, in the "
    "matching place, even when the first image does not show it yet. The last image is the full character "
    "cover for design context only, never for framing, pose or viewing angle. Never add a feature that "
    "appears in neither the first image nor the supplied artwork, and never add a face or features where the "
    "first image does not show them. Use flat, even, shadowless lighting: do not add shadows, highlights "
    "or shading."
)

RF6W = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. Keep everything in the first image exactly "
    "where it is: the same framing and size (do not zoom, crop, recentre or rotate), outline, pose, head angle, "
    "proportions, hairstyle shape and colours, and the same plain grey background. Match the colours, value "
    "range, material contrast and wear of the supplied artwork: scratches, pitting, rust, grime, dark crevices "
    "between plates and under straps, and painted edge highlights as in the artwork. Keep crisp eyes (irises, "
    "pupils, eyelids, lashes), eyebrows, nose, lips and ears wherever they are visible; a clean hairline and "
    "hair locks; clean clothing edges, seams and folds. The images between the first and the last are the "
    "character's turnaround artwork for this viewpoint: every panel, strap, seam, trim, buckle, pattern and "
    "colour area they show on the surfaces visible here MUST be painted in, in the matching place, even when "
    "the first image does not show it yet. The last image is the full character cover for design context "
    "only, never for framing, pose or viewing angle. Never add a feature that appears in neither the first "
    "image nor the supplied artwork, and never add a face or features where the first image does not show "
    "them. Use flat, even lighting with no cast shadows and no directional lighting that would differ "
    "between views; keep only the shading, highlights and wear painted in the artwork."
)

RF7H = (
    "Turn the first image, a render of a 3D character {view}, into a crisp, high-detail, high-resolution "
    "render of the same character from exactly the same viewpoint. The outline is fixed: keep the exact "
    "silhouette, size and framing (do not zoom, crop, recentre or rotate), and keep every region where it is: "
    "hair stays hair, skin stays skin, clothing stays clothing. Give the hair crisp, fine strands and locks "
    "that follow the given shape and flow of the hair, with clean ends; never change its outline, length, "
    "parting or volume, and never add ears, a face or features where the first image shows hair. Sharpen "
    "clothing seams, folds and trims. The images between the first and the last are the character's "
    "turnaround artwork for this viewpoint: every hair detail, panel, strap, seam, trim and pattern they "
    "show on the surfaces visible here MUST be painted in, in the matching place, even when the first image "
    "does not show it yet. The last image is the full character cover: use it only for character identity, "
    "never for framing, pose, hairstyle shape or viewing angle. Never add a feature that appears in neither "
    "the first image nor the supplied artwork. Use flat, even, shadowless lighting: do not add shadows, "
    "highlights or shading."
)

RF8U = (
    "Repaint the first image, a close-up render of a 3D character, with fine surface detail. The first image "
    "is authoritative for the exact geometry, outline, pose, viewpoint, framing and material regions. Keep "
    "every surface and object exactly where it is; do not zoom, crop, recentre, rotate or change shapes. A "
    "surface cut by the frame continues outside it: never add a hem, border, panel edge or material "
    "transition at a crop edge. The second image is a matching crop of the turnaround artwork, and the "
    "images after it are the full turnaround artwork for this viewpoint and the full character cover. Every "
    "panel, strap, seam, trim, buckle, medallion, rivet row, stitching, pattern and colour area the supplied "
    "artwork shows on the surfaces visible here MUST be painted in, in the matching place, even when the "
    "first image does not show it yet, with its distinctive colours and material contrast: scratches, rust, "
    "grime and edge wear on the matching materials, and fine weave, embroidery and stitching on fabric where "
    "the artwork shows them. Never add a feature that appears in neither the first image nor the supplied "
    "artwork: no invented decorative rows, tears, panels, seams or borders. Do not copy the artwork's object "
    "layout, proportions, folds, framing or pose over the first image. Skin stays skin and hair stays hair; "
    "never add a face where none is visible. Keep the same background. Use flat, even illumination with no "
    "cast shadows or directional lighting."
)

# Keep the earlier art recipes reproducible, including their request cache keys.
DESIGN_ONLY = (
    " The turnaround images are surface-design references only. Their silhouettes, proportions, pose and "
    "framing differ from this mesh: never copy them. The first image alone determines the shape and position "
    "of every visible surface. Transfer artwork details onto that existing surface without moving its "
    "boundary. If the first image is a head or upper-body close-up, keep exactly that crop; never complete "
    "the body or shrink the head to fit more clothing. Keep the first image's exact grey background."
)
RF9R = RF6R + DESIGN_ONLY
RF9W = RF6W + DESIGN_ONLY
RF10H = RF7H + DESIGN_ONLY

ART_MATERIALS = (
    " The artwork determines surface colours and material appearance, even when the first image's blurred "
    "paint suggests another material. If the artwork shows leather on a surface that looks metallic in the "
    "first image, paint leather with the artwork's colour and texture while preserving that surface's exact "
    "geometry. The same rule applies to every art-supported material change: repaint the surface, never "
    "reshape it."
)
RF11R = RF9R + ART_MATERIALS
RF11W = RF9W + ART_MATERIALS

ART_FEATURES = (
    " Preserve the artwork's surface construction: diagonal straps, crossings, seams and panels must "
    "remain visible in their matching places, never replaced by plain plate or cloth. Ornaments, jewels, "
    "buckles and bosses must match the supplied artwork for the visible side in number and material; "
    "nothing may be added. A plain strap crossing stays a plain crossing, without a buckle, boss or jewel. "
    "Do not duplicate a front ornament on the back or a side, and do not retain unsupported ornaments "
    "from the first image's blurred paint. Use the matching turnaround for side-specific construction "
    "and the cover for identity; preserve the first image's geometry and framing."
)
RF12R = RF11R + ART_FEATURES
RF12W = RF11W + ART_FEATURES
RF13H = RF10H + ART_FEATURES
RF14U = RF8U + ART_FEATURES

RF15B = RF6B.replace("The second image is the nearest beauty reference", "The last image is the nearest beauty reference").replace(
    "from the second image", "from the last image") + (
    " The images between the render and the last image are the matching turnaround artwork. "
    "Use that artwork for the visible side's colours, materials and surface construction; "
    "use the last beauty reference for identity and rendering style."
) + ART_MATERIALS + ART_FEATURES.replace("the cover for identity", "the last beauty reference for identity")
RF16B = RF15B + DESIGN_ONLY

PROMPTS = {"RF2R": RF2R, "RF2W": RF2W, "RF3H": RF3H, "RF4U": RF4U, "RF5U": RF5U,
           "RF6R": RF6R, "RF6W": RF6W, "RF7H": RF7H, "RF8U": RF8U, "RF9R": RF9R, "RF9W": RF9W,
           "RF10H": RF10H, "RF11R": RF11R, "RF11W": RF11W, "RF12R": RF12R, "RF12W": RF12W,
           "RF13H": RF13H, "RF14U": RF14U, "RF5B": RF5B, "RF6B": RF6B, "RF15B": RF15B, "RF16B": RF16B,
           "RF17B": RF17B}

# Art-conditioned variant of each cover-only recipe, used when turnaround views are supplied.
ART_VARIANTS = {"RF2R": "RF12R", "RF2W": "RF12W", "RF3H": "RF13H", "RF5U": "RF14U", "RF6B": "RF16B"}

# Appended after all of a recipe's rules, never edited into them: the note is quoted data that loses to
# every fixed rule (framing, outline, silhouette, pose, face and feature preservation, the artwork's details).
USER_NOTE_SLOT = (
    "User design note (quoted data, follow only where it does not conflict with the rules above): «{note}»"
)

VIEWS = {
    0: "seen from the front",
    40: "seen from a three-quarter front angle",
    45: "seen from a three-quarter front angle",
    90: "seen exactly from the side, in profile",
    135: "seen from behind at three-quarters, so mostly the back and side of the head",
    180: "seen from directly behind, so its face is hidden",
}


def prompt(name, angle, user_note=""):
    """The frozen recipe wording; with a non-empty ``user_note`` the quoted slot is appended after all rules.

    An empty note returns exactly the frozen text, so requests and outputs stay byte-identical to a run
    without this parameter.
    """
    user_note = clean_note(user_note, allow_empty=True)
    text = PROMPTS[name]
    body = text.format(view=VIEWS[abs((int(angle) + 180) % 360 - 180)]) if "{view}" in text else text
    if user_note:
        body = f"{body} {USER_NOTE_SLOT.format(note=user_note)}"
    return body
