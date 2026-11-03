import unittest

from herogen.service.web import roblox_triangle_counts


class RobloxTriangleCounts(unittest.TestCase):
    def test_reads_both_levels_of_detail(self):
        rec = {'body': {'lods': {'high': {'triangles': 22736, 'budget': 24000},
                                 'low': {'triangles': 5818, 'budget': 6000}}}}
        self.assertEqual(roblox_triangle_counts(rec),
                         {'high': 22736, 'high_budget': 24000, 'low': 5818, 'low_budget': 6000})

    def test_a_record_without_counts_gives_none(self):
        self.assertIsNone(roblox_triangle_counts(None))
        self.assertIsNone(roblox_triangle_counts({'body': {'lods': {'high': {'triangles': 1, 'budget': 2}}}}))
        self.assertIsNone(roblox_triangle_counts({'body': {'lods': {'high': {'triangles': True, 'budget': 2},
                                                                    'low': {'triangles': 1, 'budget': 2}}}}))


class RefineSuggestions(unittest.TestCase):
    def test_keeps_only_each_steps_short_text(self):
        from herogen.service.web import refine_suggestions
        data = {'source': 'spark', 'texture': {'text': ' Make the lenses\n amber. ', 'basis': 'mismatch'},
                'turnaround': {'text': 'x' * 201}, 'error': 'ApiError: secret detail'}
        self.assertEqual(refine_suggestions(data), {'texture': 'Make the lenses amber.'})
        self.assertIsNone(refine_suggestions({'texture': {'text': ''}}))
        self.assertIsNone(refine_suggestions([]))


if __name__ == '__main__':
    unittest.main()
