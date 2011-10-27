try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.derived import KeyPointValue, KeyPointValueNode

class TestAbstractNode(unittest.TestCase):
    
    def test_node(self):
        pass
    

class TestKeyPointValueNode(unittest.TestCase):
    
    def test_create_kpv(self):
        """ Tests name format substitution and return type
        """
        KPV = type('kpv', (KeyPointValueNode,), dict(derive=lambda x:x))
        knode = KPV()
        knode.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)dft'
        knode.RETURN_OPTIONS = {'phase':['ascent', 'descent'],
                                'altitude':[1000,1500],
                                }
        # use keyword arguments
        spd_kpv = knode.create_kpv(10, 12.5, phase='descent', altitude=1000.0)
        self.assertTrue(isinstance(spd_kpv, KeyPointValue))
        self.assertEqual(spd_kpv.index, 10)
        self.assertEqual(spd_kpv.value, 12.5)
        self.assertEqual(spd_kpv.name, 'Speed in descent at 1000ft')
        # use dictionary argument
        # and check interpolation value 'ignored' is indeed ignored
        spd_kpv2 = knode.create_kpv(10, 12.5, dict(phase='ascent', 
                                                   altitude=1500.0,
                                                   ignored='excuseme'))
        self.assertEqual(spd_kpv2.name, 'Speed in ascent at 1500ft')
        # missing interpolation value "altitude"
        self.assertRaises(KeyError, knode.create_kpv, 1, 2,
                          dict(phase='ascent'))
        # wrong type raises TypeError
        self.assertRaises(TypeError, knode.create_kpv, 2, '3', 
                          phase='', altitude='')
        
    
    def test_generate_kpv_name_list(self):
        """ Using all RETURNS options, apply NAME_FORMAT to obtain a complete
        list of KPV names this class will create.
        """
        KPV = type('kpv', (KeyPointValueNode,), dict(derive=lambda x:x))
        knode = KPV()
        knode.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
        knode.RETURN_OPTIONS = {'altitude' : range(100, 701, 300),'phase' : ['ascent', 'descent']}
        kpv_names = knode.kpv_names()
        
        self.assertEqual(kpv_names, ['Speed in ascent at 100 ft',
                                     'Speed in ascent at 400 ft',
                                     'Speed in ascent at 700 ft',
                                     'Speed in descent at 100 ft',
                                     'Speed in descent at 400 ft',
                                     'Speed in descent at 700 ft',
                                     ])
        
    def test_validate_name(self):
        """ Ensures that created names have a validated option
        """
        KPV = type('kpv', (KeyPointValueNode,), dict(derive=lambda x:x))
        knode = KPV()
        knode.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
        knode.RETURN_OPTIONS = {'altitude' : range(100,1000,100),
                                'phase' : ['ascent', 'descent']}
        self.assertTrue(knode._validate_name('Speed in ascent at 500 ft'))
        self.assertTrue(knode._validate_name('Speed in descent at 900 ft'))
        self.assertTrue(knode._validate_name('Speed in descent at 100 ft'))
        self.assertRaises(ValueError, knode._validate_name, 'Speed in ascent at -10 ft')
    
        
    ##def test_kpv_names_exist_on_db(self):
        ###urr, this probably shouldn't be here!
        ##pass
    