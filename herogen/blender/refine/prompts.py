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

PROMPTS = {"RF2R": RF2R, "RF3H": RF3H}

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
