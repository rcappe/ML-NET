using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace ml_net_get_started
{
    class Program
    {

        static void Main(string[] args)
        {
            var types = AppDomain.CurrentDomain.GetAssemblies()
                        .SelectMany(s => s.GetTypes())
                        .Where(p => typeof(ITask).IsAssignableFrom(p) && p.IsClass)
                        .OrderBy(o => o.Name)
                        .ToList();
            Help(types);

            string value;
            do
            {
                value = Console.ReadLine();

                int index;
                if (int.TryParse(value, out index))
                {
                    var instance = Activator.CreateInstance(types[index]);
                    MethodInfo method = types[index].GetMethod("Run");
                    method.Invoke(instance, new object[] { });
                }

            } while (value != "q");

        }

        public static void Help(List<Type> types)
        {
            Console.WriteLine(@"Which task do you like to run?");

            for (var i = 0; i < types.Count; i++)
            {
                Console.WriteLine(@"{0}) {1}", i, types[i].Name);
            }

            Console.WriteLine(@"q) Exit");
        }
    }

    public interface ITask
    {
        void Run();
    }
}
